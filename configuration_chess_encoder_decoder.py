from transformers import PretrainedConfig
import torch

class ChessModelConfig(PretrainedConfig):
    model_type = "chess-model"

    def __init__(
        self,
        vocab_size=500,
        hidden_size=768,
        num_attention_heads=12,
        num_encoder_layers=6,
        num_decoder_layers=6,
        intermediate_size=3072,
        hidden_act="silu",
        prefix_length=10,
        max_position_embeddings=2048,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-6,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        use_regression=True,
        regression_weight=1.0,
        tie_word_embeddings=True,
        **kwargs
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs
        )
        
        # Architecture sizes
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.intermediate_size = intermediate_size
        self.prefix_length = prefix_length
        self.max_position_embeddings = max_position_embeddings
        
        # Dropout and regularization
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        
        # Activation function
        self.hidden_act = hidden_act
        
        # Model behavior
        self.use_regression = use_regression
        self.regression_weight = regression_weight
        self.tie_word_embeddings = tie_word_embeddings

        # Computed attributes
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size {self.hidden_size} is not divisible by the number of attention heads {self.num_attention_heads}"
            )
        
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        
    def get_attention_mask(self, attention_mask):
        """Convert attention mask to causal mask scores"""
        batch_size, seq_length = attention_mask.size()
        # Create causal mask
        # [batch_size, 1, seq_length, seq_length]
        causal_mask = torch.triu(
            torch.ones((seq_length, seq_length), dtype=torch.bool),
            diagonal=1,
        ).unsqueeze(0).unsqueeze(1)
        
        # Convert attention_mask to float and unsqueeze
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        attention_mask = (1.0 - attention_mask) * torch.finfo(torch.float32).min
        
        # Combine causal and attention masks
        mask = causal_mask | (attention_mask == torch.finfo(torch.float32).min)
        return mask.to(attention_mask.device)