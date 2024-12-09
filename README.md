# chess-custom-encoder-decoder

## **Overview**
This project focuses on building a custom encoder-decoder Transformer-based model specialized for chess-related tasks. By leveraging large-scale chess datasets (e.g., from **Lichess** and **LAION**), the model can learn to:

- Predict moves in a given board position (move prediction).
- Solve chess puzzles by generating the correct move sequence (puzzle-solving).
- Evaluate board positions by providing centipawn scores or mate predictions (position evaluation).

To handle the unique characteristics of chess data, the project implements a **custom tokenizer**, a **chess-specific data collator**, and harnesses **Hugging Face’s streaming dataset** capabilities to efficiently deal with large, heterogeneous datasets.

## **Key Features**
- **Custom Tokenization** for Chess:
  - Encodes FEN (Forsyth–Edwards Notation) into a sequence of tokens.
  - Transforms move lists into tokenized sequences, including promotions.
  - Incorporates special tokens for handling metadata like castling rights and side-to-move.
  
- **Mixed Data Collation**:
  - Dynamically batches examples of different types (full games, puzzles, position evaluations).
  - Properly pads sequences and sets regression targets (for position evaluations) in a single unified batch.

- **Streaming Datasets**:
  - Efficiently handle large-scale data from multiple sources using Hugging Face’s `datasets` library in streaming mode.
  - Interleave datasets with custom probabilities to achieve balanced training across different tasks.

## **Project Structure**
- **`configuration_chess_encoder_decoder.py`**: Defines a custom configuration class (`ChessModelConfig`) for the model.
- **`modeling_chess_encoder_decoder.py`**: Implements the model architecture, including:
  - A FenEncoder for processing board states.
  - A PrefixProjector to create a prefix context for the decoder.
  - A MoveDecoder for predicting moves or related output sequences.
  - A regression head for position evaluations.

- **`chess_tokenizer.py`**: A custom tokenizer to handle FEN and move encoding.
- **`chess_mix_objective_collator.py`**: A collator that merges different example types (game, puzzle, analysis) into a single batch.
- **`no_trainer.py`**: A no-trainer style script (inspired by the Hugging Face no-trainer example) that:
  - Loads streaming datasets.
  - Interleaves multiple datasets.
  - Sets up the data loaders, model, optimizer, and runs training and evaluation loops.

- **`requirements.txt`**: Lists the required Python packages.

## **Data Input Format**
This project relies on three main datasets, each with a distinct format:

1. **LAION/Strategic Game Chess**:
   - **Type**: Full chess games.
   - **Fields**:
     - `fen`: Optional starting FEN position. Defaults to the standard initial chess position if not provided.
     - `moves`: A list of move strings (e.g., `["e2e4", "e7e5", "g1f3"]`).
   - **Use Case**: Training the model to predict moves from any position within a full game.

2. **Lichess/Chess Puzzles**:
   - **Type**: Chess puzzles focused on tactics.
   - **Fields**:
     - `fen`: The FEN starting position of the puzzle.
     - `moves`: The correct solution moves for the puzzle.
   - **Use Case**: Puzzle-solving by predicting the correct sequence of moves.

3. **Lichess/Chess Position Evaluations**:
   - **Type**: Engine evaluations of board states.
   - **Fields**:
     - `fen`: The position in FEN format.
     - `cp`: Centipawn evaluation (positive = advantage White, negative = advantage Black).
     - `mate`: Number of moves until mate (positive = White mating, negative = Black mating).
     - `moves`: The principal variation (best line of moves) from the given position.
   - **Use Case**: Predicting position evaluations and best moves from a given FEN.

**Hugging Face Dataset Paths**:
- `laion/strategic_game_chess`
- `Lichess/chess-puzzles`
- `Lichess/chess-position-evaluations`

## **Data Processing Steps**
1. **Tokenizer** (`ChessTokenizer`):
   - **FEN Encoding**: Breaks down the FEN string into pieces, castling rights, en passant targets, and move counters.
   - **Move Encoding**: Splits moves into tokens representing origin and destination squares, plus promotion indicators if any.
   - **Special Tokens**: `<START>`, `<END>`, `<PAD>`, `<MASK>`, `<RANKSEP>`, `<WHITE_TO_MOVE>`, `<BLACK_TO_MOVE>` to handle chess-specific semantics and sequence manipulation.
   - **Example**:
     - FEN: `"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"`
     - Moves: `["e2e4", "e7e5"]`

2. **Data Collator** (`MixedDataCollator`):
   - Handles batching and padding for varied lengths.
   - Interleaves different example types:
     - **Full Game**: Randomly selects a subsequence of moves.
     - **Puzzle**: Uses FEN and solution moves directly.
     - **Analysis**: Includes regression targets for `cp` or `mate` evaluations.
   - Outputs a dictionary with:
     - `fen_input_ids`, `fen_attention_mask`: For the encoder input (FEN).
     - `decoder_input_ids`, `labels`: For the decoder side (move prediction).
     - `regression_labels`: Numerical evaluation scores (if available).
  
   **Example batch format**:
   ```python
   {
     "fen_input_ids": [[101, 23, 5, ...], [101, 24, 6, ...]],
     "fen_attention_mask": [[1, 1, 1, ...], [1, 1, 0, ...]],
     "decoder_input_ids": [[202, 303, ...], [202, 304, ...]],
     "labels": [[303, ...], [304, ...]],
     "regression_labels": [30.0, -100000.0],
   }
   ```

3. **Dataset Streaming & Interleaving**:
   - Uses `datasets.load_dataset` with `streaming=True`.
   - Maps each dataset to a standardized format (full_game, puzzle, analysis).
   - Interleaves datasets with specified probabilities (e.g., 40% full_game, 40% puzzle, 20% analysis).
   
   **Example**:
   ```python
   from datasets import load_dataset, interleave_datasets

   strategic_game_ds = load_dataset("laion/strategic_game_chess", split="train", streaming=True)
   puzzles_ds = load_dataset("Lichess/chess-puzzles", split="train", streaming=True)
   analysis_ds = load_dataset("Lichess/chess-position-evaluations", split="train", streaming=True)

   # Map to standardized format
   strategic_game_mapped = strategic_game_ds.map(map_strategic_game)
   puzzles_mapped = puzzles_ds.map(map_puzzle)
   analysis_mapped = analysis_ds.map(map_analysis)

   # Interleave
   combined_dataset = interleave_datasets(
       [strategic_game_mapped, puzzles_mapped, analysis_mapped],
       probabilities=[0.4, 0.4, 0.2],
       seed=42
   )
   ```

## **Model Architecture**
1. **FenEncoder**:
   - A stack of Transformer encoder layers that process the tokenized FEN representation.
   - Outputs a contextual embedding of the chess position.

2. **PrefixProjector**:
   - Takes the encoder output and projects a summary embedding into a prefix sequence.
   - The prefix is prepended to the decoder input to guide the move generation or evaluation.

3. **MoveDecoder**:
   - A Transformer decoder stack that attends to both the prefix states and the previously generated tokens.
   - Predicts the next token in a move sequence.

4. **Regression Head**:
   - A linear layer on top of the encoder's pooled output to predict continuous values like centipawn scores or mate distances.

**Configuration:**
- Set via `ChessModelConfig`, allowing customization of:
  - Hidden sizes, number of attention heads, number of layers.
  - Activation functions, dropout rates.
  - Regression usage and weighting.

## **Training**
The `no_trainer.py` script demonstrates how to train the model using `Accelerate` without a formal training loop from Transformers’ `Trainer`:

- **Prerequisites**:
  - Install dependencies: `pip install -r requirements.txt`
  - Ensure you have access to the `laion/strategic_game_chess`, `Lichess/chess-puzzles`, and `Lichess/chess-position-evaluations` datasets.
  - Configure your environment for `wandb` logging if desired.

- **Running the Training**:
  ```bash
  python no_trainer.py
  ```

- **Script Flow**:
  1. **Load Datasets**: Streams the datasets, maps, and interleaves them.
  2. **Create Dataloaders**: Builds training and evaluation DataLoaders.
  3. **Initialize Model & Config**: Sets up the `ChessModel` with custom config.
  4. **Train**: Runs a training loop, periodically evaluating on a holdout set.
  5. **Checkpoints**: Saves model checkpoints after certain intervals.

- **Logs & Checkpoints**:
  - Training and evaluation losses are logged to `wandb` if configured.
  - Model checkpoints are saved in `./checkpoints` by default.

## **Evaluation**
- The script evaluates the model at regular intervals during training.
- Evaluation loss on the holdout set provides a rough measure of performance.
- For deeper analysis, consider:
  - Generating moves from given FEN positions.
  - Comparing predicted evaluations (cp or mate) against engine evaluations.
  - Testing puzzle accuracy.

## **Applications**
1. **Move Prediction**:
   - Given a board state (FEN), the model predicts the most likely next move(s).
   - Can be used to mimic typical chess engine recommendations or human-like move suggestions.

2. **Puzzle Solving**:
   - Input the FEN of a puzzle.
   - The model outputs the correct solution moves.
   - Useful for automatically solving or generating chess tactics.

3. **Position Evaluation**:
   - Directly predict centipawn or mate scores for a given FEN.
   - This can augment or replace traditional chess engines in evaluation tasks, though model performance would depend on training quality and data coverage.

## **Future Work**
- **Model Scaling**: Experiment with larger models or integrate advanced architectures (e.g., GPT-style decoders).
- **Integration with Chess Engines**: Use the model as a heuristic guide or hybrid component in an engine-driven pipeline.
