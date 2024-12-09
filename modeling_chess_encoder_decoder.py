import math
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from torch.nn import CrossEntropyLoss, MSELoss
from configuration_chess_encoder_decoder import ChessModelConfig

def _make_causal_mask(input_ids_shape, device):
    batch_size, target_length = input_ids_shape
    mask = torch.triu(torch.ones((target_length, target_length), device=device), diagonal=1).bool()
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
    return mask

class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_position_embeddings = max_position_embeddings
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_len=None):
        if seq_len > self.max_position_embeddings:
            raise ValueError(f"Sequence length {seq_len} greater than maximum length {self.max_position_embeddings}")
        
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.cos_cached = emb.cos()[None, None, :, :]
            self.sin_cached = emb.sin()[None, None, :, :]
        return self.cos_cached, self.sin_cached

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x

class ChessAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings
        )

    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_length, _ = hidden_states.shape
        
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(q, seq_length)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling

        if attention_mask is not None:
            scores = scores + attention_mask

        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_length, self.hidden_size)
        
        return self.o_proj(attn_output)

class ChessMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)

class FenEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': ChessAttention(config),
                'attention_norm': RMSNorm(config.hidden_size),
                'mlp': ChessMLP(config),
                'mlp_norm': RMSNorm(config.hidden_size)
            })
            for _ in range(config.num_encoder_layers)
        ])
        self.final_norm = RMSNorm(config.hidden_size)
        
    def forward(self, input_ids, attention_mask=None):
        hidden_states = self.embed(input_ids)

        # Convert attention_mask [B, S] -> [B, 1, 1, S]
        if attention_mask is not None:
            # Convert boolean or int mask to float: 1 for keep, 0 for mask
            attention_mask = attention_mask[:, None, None, :].float()  
            # Convert to large negative values for masked positions
            attention_mask = (1.0 - attention_mask) * torch.finfo(torch.float32).min

        for layer in self.layers:
            # Pre-norm architecture
            attn_output = layer['attention'](
                layer['attention_norm'](hidden_states),
                attention_mask
            )
            hidden_states = hidden_states + attn_output
            
            mlp_output = layer['mlp'](layer['mlp_norm'](hidden_states))
            hidden_states = hidden_states + mlp_output
        
        hidden_states = self.final_norm(hidden_states)
        return hidden_states


class PrefixProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.prefix_length = config.prefix_length
        self.projector = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 2),
            ACT2FN[config.hidden_act],
            RMSNorm(config.hidden_size * 2),
            nn.Linear(config.hidden_size * 2, config.hidden_size)
        )
        
    def forward(self, encoder_hidden):
        # Average pool the encoder hidden states
        pooled = encoder_hidden.mean(dim=1)  # [batch, hidden_size]
        # Project to decoder hidden size
        projected = self.projector(pooled)  # [batch, hidden_size]
        # Expand to prefix length
        prefix = projected.unsqueeze(1).expand(-1, self.prefix_length, -1)
        return prefix

class MoveDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': ChessAttention(config),
                'attention_norm': RMSNorm(config.hidden_size),
                'mlp': ChessMLP(config),
                'mlp_norm': RMSNorm(config.hidden_size)
            })
            for _ in range(config.num_decoder_layers)
        ])
        self.final_norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
    def forward(self, input_ids, prefix_states, attention_mask=None):
        # Embed decoder inputs
        hidden_states = self.embed(input_ids)
        
        # Concatenate prefix states with embedded inputs
        hidden_states = torch.cat([prefix_states, hidden_states], dim=1)
        
        # Create causal mask for the combined sequence [B, S, S]
        causal_mask = _make_causal_mask(hidden_states.shape[:2], hidden_states.device)
        causal_mask = causal_mask.unsqueeze(1).float()  # [B, 1, S, S]
        causal_mask = (1.0 - causal_mask) * torch.finfo(torch.float32).min

        if attention_mask is not None:
            # Adjust decoder attention mask as well [B, S]
            full_seq_length = hidden_states.size(1)
            prefix_length = prefix_states.size(1)

            # The attention mask for the decoder inputs (excluding prefix)  
            # should still be [B, S_decoder]. We'll create a combined mask.
            # First, create a mask for the prefix tokens (all visible):
            prefix_mask = torch.ones(
                attention_mask.size(0), prefix_length, device=attention_mask.device, dtype=attention_mask.dtype
            )

            # Combine prefix_mask and attention_mask
            combined_mask = torch.cat([prefix_mask, attention_mask], dim=1)  # [B, S_total]

            # Convert to [B, 1, 1, S_total] with large negative values for masked positions
            combined_mask = combined_mask[:, None, None, :].float()
            combined_mask = (1.0 - combined_mask) * torch.finfo(torch.float32).min

            # Combine with causal_mask (they should both be [B, 1, S, S])
            # Note: If you're using causal + padding, ensure compatibility. Typically, you'd add them:
            causal_mask = causal_mask + combined_mask

        for layer in self.layers:
            attn_output = layer['attention'](
                layer['attention_norm'](hidden_states),
                causal_mask
            )
            hidden_states = hidden_states + attn_output

            mlp_output = layer['mlp'](layer['mlp_norm'](hidden_states))
            hidden_states = hidden_states + mlp_output

        hidden_states = self.final_norm(hidden_states)

        # Return only the logits for the actual sequence (excluding prefix)
        seq_length = input_ids.size(1)
        logits = self.lm_head(hidden_states[:, -seq_length:])
        return logits


# In modeling_chess_encoder_decoder.py
class ChessModel(PreTrainedModel):
    config_class = ChessModelConfig
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.fen_encoder = FenEncoder(config)
        self.prefix_projector = PrefixProjector(config)
        self.move_decoder = MoveDecoder(config)
        self.regression_head = nn.Linear(config.hidden_size, 1)
        self.post_init()

    def forward(
        self,
        fen_input_ids,
        fen_attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None
    ):
        # Encode FEN
        encoder_hidden = self.fen_encoder(fen_input_ids, fen_attention_mask)
        # Generate prefix states
        prefix_states = self.prefix_projector(encoder_hidden)
        # Get regression predictions from encoder outputs
        regression_preds = self.regression_head(encoder_hidden.mean(dim=1))
        # Decode moves with prefix
        logits = self.move_decoder(decoder_input_ids, prefix_states, decoder_attention_mask)
        
        # Return only predictions, no loss
        return {
            "logits": logits,
            "regression_preds": regression_preds
        }

    @staticmethod
    def calculate_decoder_loss(logits, labels, pad_token_id=-100):
        loss_fct = CrossEntropyLoss(ignore_index=pad_token_id, reduction='sum')
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        return loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    @staticmethod
    def calculate_regression_loss(regression_preds, regression_labels, regression_mask):
        # Compute MSE only for masked positions
        mse = (regression_preds.squeeze(-1) - regression_labels) ** 2
        mse = mse * regression_mask
        denom = regression_mask.sum() + 1e-8
        return mse.sum() / denom
