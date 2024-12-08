# chess-custom-encoder-decoder

## **Project Overview**

This project aims to create a deep learning model tailored for chess-related tasks, such as move prediction, puzzle-solving, or evaluating board positions. It leverages datasets from platforms like **Lichess** and **LAION**, along with custom tokenizer and collator implementations to handle chess-specific data, such as Forsyth-Edwards Notation (FEN), move sequences, and board evaluations.

The model processes chess data using:
- **Custom tokenization**: Specialized for chess board states and moves.
- **Data collation**: Merges examples into batches while handling variable sequence lengths and padding.
- **Streaming datasets**: Efficiently processes large-scale data using Hugging Face’s dataset streaming capabilities.

---

## **Data Input Format**

The project uses three primary datasets, each with unique chess-specific data:

### **1. Datasets**
#### **a. LAION/Strategic Game Chess**
- **Type**: Full games.
- **Fields**:
  - `fen`: Optional starting position in FEN format (default to standard starting position if not provided).
  - `moves`: List of moves (e.g., `["e2e4", "e7e5", "g1f3"]`).
- **Use Case**: Predict moves from a starting position or anywhere in the game's history.

#### **b. Lichess/Chess Puzzles**
- **Type**: Chess puzzles (tactics).
- **Fields**:
  - `fen`: Starting position for the puzzle in FEN format.
  - `moves`: List of correct moves to solve the puzzle.
- **Use Case**: Solve tactical positions by predicting a sequence of moves.

#### **c. Lichess/Chess Position Evaluations**
- **Type**: Board evaluations from chess engines.
- **Fields**:
  - `fen`: Position in FEN format.
  - `cp`: Centipawn evaluation (e.g., +30 for slight White advantage).
  - `mate`: Moves to mate (positive for White, negative for Black).
  - `moves`: Principal variation from the position (e.g., `["e2e4", "e7e5", "g1f3"]`).
- **Use Case**: Evaluate board positions or predict principal variations.


**Datasets huggingface paths:**
- laion/strategic_game_chess
- Lichess/chess-puzzles
- Lichess/chess-position-evaluations
---

## **Data Processing**

### **1. Tokenizer**
The **ChessTokenizer** is designed to tokenize FEN strings, chess moves, and other chess-specific data into numerical representations suitable for neural networks.

#### **Key Features**
1. **FEN Encoding**:
   - Converts a FEN string into tokens that represent the board state, side to move, castling rights, en passant target, and counters.
   - Example: `"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"`
     - Tokenized into a sequence of tokens for pieces, ranks, and additional state.

2. **Move Encoding**:
   - Converts a list of chess moves (e.g., `["e2e4", "e7e5"]`) into tokens representing source and destination squares, along with promotion types if applicable.

3. **Special Tokens**:
   - Adds tokens like `<START>`, `<END>`, `<PAD>`, and `<MASK>` for handling sequences.
   - FEN-specific tokens like `<RANKSEP>` and `<WHITE_TO_MOVE>` are also included.

4. **Board Visualization**:
   - Optionally generates a PNG image of the chessboard for debugging or visualization.

---

### **2. Data Collator**
The **MixedDataCollator** processes examples into a batch format suitable for training.

#### **Key Responsibilities**
1. **Example Types**:
   - **Full Game**: Randomly samples a position from the game's history.
   - **Puzzle**: Uses the FEN and solution moves.
   - **Analysis**: Encodes evaluation data (`cp`, `mate`) and principal variations.

2. **Collation**:
   - Converts input FEN and move sequences into token IDs.
   - Pads sequences to the same length within a batch.
   - Handles additional features like regression labels for centipawn evaluations.

3. **Output Batch Format**:
   - `fen_input_ids`: Tokenized FEN IDs for encoder input.
   - `fen_attention_mask`: Mask for padding in FEN sequences.
   - `decoder_input_ids`: Tokenized move IDs for decoder input.
   - `labels`: Target move IDs for computing loss.
   - `regression_labels`: Evaluation scores (centipawn or mate) if applicable.

#### **Example Output**
```python
{
    "fen_input_ids": [[101, 23, 5, ...], [101, 24, 6, ...]],  # Batched FEN sequences
    "fen_attention_mask": [[1, 1, 1, ...], [1, 1, 0, ...]],  # Padding masks
    "decoder_input_ids": [[202, 303, 15, ...], [202, 304, 16, ...]],  # Batched moves
    "labels": [[303, 15, ...], [304, 16, ...]],  # Target moves for loss calculation
    "regression_labels": [30.0, -100000.0],  # Example evaluations (centipawn or mate)
}
```

---

### **3. Dataset Streaming**
The Hugging Face `datasets` library is used to stream large datasets, combining multiple sources into a single unified dataset.

#### **Implementation**
- **Mapping**:
  - Each dataset is mapped into a standardized format with `example_type` (e.g., `full_game`, `puzzle`, or `analysis`) and its relevant fields.
- **Interleaving**:
  - Combines datasets in a probabilistic manner to balance different types of training examples.

#### **Example**
```python
from datasets import load_dataset, interleave_datasets

# Load datasets in streaming mode
strategic_game_ds = load_dataset("laion/strategic_game_chess", split="train", streaming=True)
puzzles_ds = load_dataset("Lichess/chess-puzzles", split="train", streaming=True)
analysis_ds = load_dataset("Lichess/chess-position-evaluations", split="train", streaming=True)

# Map to standardized format
strategic_game_mapped = strategic_game_ds.map(map_strategic_game)
puzzles_mapped = puzzles_ds.map(map_puzzle)
analysis_mapped = analysis_ds.map(map_analysis)

# Interleave datasets
combined_dataset = interleave_datasets(
    [strategic_game_mapped, puzzles_mapped, analysis_mapped],
    probabilities=[0.4, 0.4, 0.2],
    seed=42
)
```

---

## **Applications**

1. **Move Prediction**:
   - Predict the best move given a board state, mimicking human play or chess engine recommendations.

2. **Puzzle Solving**:
   - Train the model to solve tactical puzzles by predicting a sequence of moves.

3. **Position Evaluation**:
   - Provide centipawn scores or mate evaluations for arbitrary board positions.
