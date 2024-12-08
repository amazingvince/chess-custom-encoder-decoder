import os
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers import get_scheduler
from chess_tokenizer import ChessTokenizer
from chess_mix_objective_collator import MixedDataCollator
from modeling_chess_encoder_decoder import ChessModel
from configuration_chess_encoder_decoder import ChessModelConfig
from datasets import load_dataset, interleave_datasets, Dataset

import torch._dynamo
torch._dynamo.config.suppress_errors = True


EXAMPLE_TYPES = {
    'FULL_GAME': 'full_game',
    'PUZZLE': 'puzzle',
    'ANALYSIS': 'analysis'
}

def map_strategic_game(example):
    moves = []
    if 'Moves' in example:
        if isinstance(example['Moves'], str):
            moves = example['Moves'].split()
        elif isinstance(example['Moves'], list):
            moves = example['Moves']

    return {
        "example_type": EXAMPLE_TYPES['FULL_GAME'],
        "fen": example.get("fen", None),
        "Moves": moves
    }

def map_puzzle(example):
    moves = []
    if 'Moves' in example:
        if isinstance(example['Moves'], str):
            moves = example['Moves'].split()
        elif isinstance(example['Moves'], list):
            moves = example['Moves']

    return {
        "example_type": EXAMPLE_TYPES['PUZZLE'],
        "fen": example.get("FEN", None),
        "Moves": moves
    }

def map_analysis(example):
    line = []
    if 'line' in example:
        if isinstance(example['line'], str):
            line = example['line'].split()
        elif isinstance(example['line'], list):
            line = example['line']

    return {
        "example_type": EXAMPLE_TYPES['ANALYSIS'],
        "fen": example.get("fen", None),
        "line": " ".join(line) if line else example.get('line', ''),
        "cp": example.get("cp", None),
        "mate": example.get("mate", None)
    }

def load_and_prepare_datasets():
    # Load datasets
    try:
        strategic_game_ds = load_dataset("laion/strategic_game_chess", split="train", streaming=True)
        print("Successfully loaded strategic game dataset")
    except Exception as e:
        print(f"Error loading strategic game dataset: {e}")
        strategic_game_ds = None

    try:
        puzzles_ds = load_dataset("Lichess/chess-puzzles", split="train", streaming=True)
        print("Successfully loaded puzzles dataset")
    except Exception as e:
        print(f"Error loading puzzles dataset: {e}")
        puzzles_ds = None

    try:
        analysis_ds = load_dataset("Lichess/chess-position-evaluations", split="train", streaming=True)
        print("Successfully loaded analysis dataset")
    except Exception as e:
        print(f"Error loading analysis dataset: {e}")
        analysis_ds = None

    datasets = []
    probabilities = []

    if strategic_game_ds is not None:
        strategic_game_mapped = strategic_game_ds.map(
            map_strategic_game,
            remove_columns=strategic_game_ds.column_names
        )
        datasets.append(strategic_game_mapped)
        probabilities.append(0.4)

    if puzzles_ds is not None:
        puzzles_mapped = puzzles_ds.map(
            map_puzzle,
            remove_columns=puzzles_ds.column_names
        )
        datasets.append(puzzles_mapped)
        probabilities.append(0.4)

    if analysis_ds is not None:
        analysis_mapped = analysis_ds.map(
            map_analysis,
            remove_columns=analysis_ds.column_names
        )
        datasets.append(analysis_mapped)
        probabilities.append(0.2)

    if probabilities:
        total = sum(probabilities)
        probabilities = [p/total for p in probabilities]

    if not datasets:
        raise ValueError("No datasets were successfully loaded")

    combined_dataset = interleave_datasets(
        datasets,
        probabilities=probabilities,
        seed=42
    )

    return combined_dataset


def create_dataloaders(tokenizer, data_collator, batch_size=8, max_steps=1000, holdout_size=500):
    # Load the streaming dataset
    combined_dataset = load_and_prepare_datasets()

    # Take first 500 samples as a holdout evaluation set
    holdout_samples = list(combined_dataset.take(holdout_size))
    holdout_dict = {key: [] for key in holdout_samples[0].keys()}
    for example in holdout_samples:
        for key, value in example.items():
            holdout_dict[key].append(value)
    eval_dataset = Dataset.from_dict(holdout_dict)

    # After taking the first 500 for eval, we skip them for training
    # NOTE: For a streaming dataset, once taken, we should re-initialize the stream
    # with a skip to continue training from after holdout samples if necessary.
    # For simplicity, let's just assume we re-load and skip:
    combined_dataset = load_and_prepare_datasets().skip(holdout_size)

    train_dataloader = DataLoader(
        combined_dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
        num_workers=0,  # For streaming dataset, use 0 or a small number
        pin_memory=False
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    return train_dataloader, eval_dataloader, combined_dataset

def evaluate(model, dataloader, accelerator):
    model.eval()
    total_loss = 0
    total_steps = 0

    with torch.no_grad():
        for batch in dataloader:
            outputs = model(
                fen_input_ids=batch["fen_input_ids"],
                fen_attention_mask=batch["fen_attention_mask"],
                decoder_input_ids=batch["decoder_input_ids"],
                decoder_attention_mask=batch["decoder_attention_mask"],
                labels=batch["labels"],  # Ensure that 'labels' is included in the batch
                regression_labels=batch.get("regression_labels", None),
                regression_mask=batch.get("regression_mask", None)
            )

            loss = outputs["loss"]  # Access the loss from the dictionary
            total_loss += loss.item()
            total_steps += 1
    
    avg_loss = total_loss / total_steps if total_steps > 0 else 0.0
    model.train()
    return avg_loss

def main():
    accelerator = Accelerator(log_with="wandb")  # Handles device, mixed precision, etc.
    accelerator.print("Setting up training...")

    tokenizer = ChessTokenizer()
    config = ChessModelConfig(
        vocab_size=tokenizer.vocab_size(),
        use_regression=True,
        hidden_size=1024,
        num_attention_heads=16,
        num_encoder_layers=14,
        num_decoder_layers=32,
        intermediate_size=4096,
        hidden_act="silu",
        prefix_length=32,
        max_position_embeddings=200,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-6,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        regression_weight=1.0,
    )
    model = ChessModel(config)

    data_collator = MixedDataCollator(
        tokenizer=tokenizer,
        max_moves=50,
        use_regression=True
    )

    # Hyperparameters
    learning_rate = 1e-4
    max_steps = 1000
    warmup_steps = 100
    batch_size = 8
    eval_every = 100
    output_dir = "./checkpoints"

    # Create dataloaders
    train_dataloader, eval_dataloader, _ = create_dataloaders(tokenizer, data_collator, batch_size=batch_size, max_steps=max_steps)

    # Optimizer and Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps
    )

    # Prepare everything with accelerator
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # Start training
    accelerator.print("Starting training...")
    global_step = 0
    model.train()

    for step, batch in enumerate(train_dataloader, start=1):
        # Forward pass
        outputs = model(
            fen_input_ids=batch["fen_input_ids"],
            fen_attention_mask=batch["fen_attention_mask"],
            decoder_input_ids=batch["decoder_input_ids"],
            decoder_attention_mask=batch["decoder_attention_mask"],
            labels=batch["labels"],  # Ensure that 'labels' is included in the batch
            regression_labels=batch.get("regression_labels", None),
            regression_mask=batch.get("regression_mask", None)
        )

        loss = outputs["loss"]  # Access the loss from the dictionary

        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        if step % 10 == 0:
            accelerator.print(f"Step {step}/{max_steps}, Loss: {loss.item():.10f}")
            accelerator.log({"train_loss": loss}, step=global_step)

        if step % eval_every == 0:
            eval_loss = evaluate(model, eval_dataloader, accelerator)
            accelerator.print(f"Step {step}: Eval Loss: {eval_loss:.10f}")
            accelerator.log({"eval_loss": loss}, step=global_step)

            # Save checkpoint (only on process 0)
            if accelerator.is_main_process:
                os.makedirs(output_dir, exist_ok=True)
                accelerator.print(f"Saving model at step {step}")
                accelerator.unwrap_model(model).save_pretrained(os.path.join(output_dir, f"checkpoint-{step}"))

        global_step += 1
        if global_step >= max_steps:
            break

    accelerator.print("Training completed!")
    accelerator.end_training()

if __name__ == "__main__":
    main()
