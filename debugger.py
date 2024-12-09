import torch
import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class TrainingMetrics:
    """Class to track training metrics"""
    train_losses: List[float] = None
    eval_losses: List[float] = None
    move_accuracies: List[float] = None
    regression_losses: List[float] = None
    learning_rates: List[float] = None
    
    def __post_init__(self):
        self.train_losses = []
        self.eval_losses = []
        self.move_accuracies = []
        self.regression_losses = []
        self.learning_rates = []
    
    def update(self, **kwargs):
        """Update metrics with new values"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                getattr(self, key).append(value)

def calculate_move_accuracy(logits: torch.Tensor, labels: torch.Tensor, pad_token_id: int = -100) -> float:
    """Calculate move prediction accuracy, ignoring padded positions"""
    predictions = logits.argmax(dim=-1)
    mask = labels != pad_token_id
    correct = ((predictions == labels) & mask).sum().item()
    total = mask.sum().item()
    return correct / total if total > 0 else 0.0

def log_gradient_stats(model) -> Dict[str, float]:
    """Calculate and format gradient statistics safely"""
    stats = {}
    total_norm = 0.0
    num_params_with_grad = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            # Calculate norm first
            param_norm = param.grad.detach().norm(2).item()
            total_norm += param_norm ** 2
            num_params_with_grad += 1
            
            # Safe calculation of mean and std
            grad_mean = param.grad.detach().mean().item()
            # Handle std calculation carefully
            if param.grad.numel() > 1:
                grad_std = param.grad.detach().std().item()
            else:
                grad_std = 0.0
                
            stats[name] = {
                "norm": param_norm,
                "mean": grad_mean,
                "std": grad_std
            }
    
    # Calculate total norm
    total_norm = np.sqrt(total_norm) if total_norm > 0 else 0.0
    stats["total"] = {
        "norm": total_norm,
        "num_params": num_params_with_grad
    }
    
    return stats

def format_gradient_stats(grad_stats: Dict[str, Dict[str, float]]) -> str:
    """Format gradient statistics for logging"""
    output = ["Gradient Statistics:"]
    output.append(f"Total norm: {grad_stats['total']['norm']:.6f}")
    output.append(f"Parameters with gradients: {grad_stats['total']['num_params']}")
    
    # Format statistics for each parameter
    for name, stats in grad_stats.items():
        if name != "total":
            output.append(f"\n{name}:")
            output.append(f"  Mean: {stats['mean']:.6f}")
            output.append(f"  Std:  {stats['std']:.6f}")
            output.append(f"  Norm: {stats['norm']:.6f}")
    
    return "\n".join(output)

def validate_batch(batch: Dict[str, torch.Tensor]) -> bool:
    """Validate batch contents and shapes"""
    required_keys = ["fen_input_ids", "fen_attention_mask", "decoder_input_ids", "labels"]
    
    # Check for required keys
    for key in required_keys:
        if key not in batch:
            print(f"Missing required key: {key}")
            return False
            
    # Check for non-empty tensors
    for key, value in batch.items():
        if value.nelement() == 0:
            print(f"Empty tensor for {key}")
            return False
            
    # Validate shapes
    batch_size = batch["fen_input_ids"].shape[0]
    for key, value in batch.items():
        if value.shape[0] != batch_size:
            print(f"Inconsistent batch size for {key}: {value.shape[0]} vs {batch_size}")
            return False
            
    return True

def log_training_step(step: int, loss: float, grad_stats: Dict[str, Dict[str, float]], 
                     pred_moves: List[str], true_moves: List[str]):
    """Format and log training step information"""
    output = [
        f"\nStep {step} Summary:",
        f"Loss: {loss:.6f}",
        # format_gradient_stats(grad_stats),
        "\nMove Predictions:",
    ]
    
    for i, (pred, true) in enumerate(zip(pred_moves[:3], true_moves[:3])):
        output.append(f"Example {i+1}:")
        output.append(f"  Predicted: {pred}")
        output.append(f"  True:      {true}")
    
    print("\n".join(output))

def decode_chess_moves(logits: torch.Tensor, labels: torch.Tensor, tokenizer) -> tuple:
    """Decode predicted and true moves, handling chess-specific tokens"""
    pred_ids = logits.argmax(dim=-1).cpu().numpy()
    label_ids = labels.cpu().numpy()
    
    pred_moves = []
    true_moves = []
    
    for pred_seq, label_seq in zip(pred_ids, label_ids):
        # Convert IDs to tokens, filtering special tokens
        pred_tokens = []
        true_tokens = []
        
        for pred_id in pred_seq:
            if pred_id in tokenizer.ids_to_tokens and pred_id not in [
                tokenizer.pad_token_id, 
                tokenizer.eos_token_id, 
                tokenizer.bos_token_id
            ]:
                pred_tokens.append(tokenizer.ids_to_tokens[pred_id])
                
        for label_id in label_seq:
            if label_id != -100 and label_id in tokenizer.ids_to_tokens:
                true_tokens.append(tokenizer.ids_to_tokens[label_id])
        
        pred_moves.append("".join(pred_tokens))
        true_moves.append("".join(true_tokens))
    
    return pred_moves, true_moves




def log_batch_stats(batch: Dict[str, torch.Tensor], tokenizer, prefix: str = ""):
    """Log detailed batch statistics with chess-specific information"""
    print(f"\n{prefix} Batch Statistics:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}:")
            print(f"  Shape: {value.shape}")
            print(f"  Type: {value.dtype}")
            print(f"  Range: [{value.min().item():.3f}, {value.max().item():.3f}]")
            print(f"  Mean: {value.float().mean().item():.3f}")
            
            # For input IDs, show sample decoded content
            if key in ["fen_input_ids", "decoder_input_ids", "labels"] and value.dim() == 2:
                sample_idx = 0
                tokens = [tokenizer.ids_to_tokens[id.item()] for id in value[sample_idx] 
                         if id.item() in tokenizer.ids_to_tokens]
                print(f"  Sample decoded tokens: {tokens[:10]}...")

def evaluate_detailed(model, dataloader, accelerator, tokenizer) -> Dict[str, float]:
    """Enhanced evaluation function with chess-specific metrics"""
    model.eval()
    total_loss = 0
    total_move_acc = 0
    total_reg_loss = 0
    num_batches = 0
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            if not validate_batch(batch):
                print("Invalid batch encountered during evaluation")
                continue
                
            outputs = model(
                fen_input_ids=batch["fen_input_ids"],
                fen_attention_mask=batch["fen_attention_mask"],
                decoder_input_ids=batch["decoder_input_ids"],
                decoder_attention_mask=batch["decoder_attention_mask"],
                labels=batch["labels"],
                regression_labels=batch.get("regression_labels"),
                regression_mask=batch.get("regression_mask")
            )
            
            loss = outputs["loss"]
            logits = outputs["logits"]
            move_acc = calculate_move_accuracy(logits, batch["labels"])
            
            # Decode moves using chess-specific method
            pred_moves, true_moves = decode_chess_moves(logits, batch["labels"], tokenizer)
            all_predictions.extend(pred_moves)
            all_labels.extend(true_moves)
            
            total_loss += loss.item()
            total_move_acc += move_acc
            if "regression_preds" in outputs:
                reg_loss = ((outputs["regression_preds"] - batch["regression_labels"]) ** 2).mean()
                total_reg_loss += reg_loss.item()
            
            num_batches += 1
    
    metrics = {
        "eval_loss": total_loss / num_batches if num_batches > 0 else float('inf'),
        "move_accuracy": total_move_acc / num_batches if num_batches > 0 else 0.0,
        "reg_loss": total_reg_loss / num_batches if num_batches > 0 else float('inf')
    }
    
    # Log some example predictions
    print("\nExample Move Predictions:")
    for pred, true in zip(all_predictions[:5], all_labels[:5]):
        print(f"Predicted: {pred}")
        print(f"True: {true}")
        print("---")
    
    return metrics

def get_gradient_stats(model):
    """Calculate gradient statistics for monitoring training"""
    total_norm = 0
    grad_stats = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            grad_stats[name] = {
                "mean": param.grad.mean().item(),
                "std": param.grad.std().item(),
                "norm": param_norm.item()
            }
    
    total_norm = total_norm ** 0.5
    grad_stats["total_norm"] = total_norm
    
    return grad_stats

