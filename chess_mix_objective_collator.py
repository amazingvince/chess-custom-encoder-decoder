from dataclasses import dataclass
from typing import List, Dict, Any
import torch
import chess
from chess_tokenizer import ChessTokenizer

@dataclass
class MixedDataCollator:
    tokenizer: "ChessTokenizer"
    max_moves: int = 50
    use_regression: bool = True
    debug: bool = True  # Set to True to enable debug printing

    def get_moves(self, example: Dict[str, Any]) -> List[str]:
        """Return a list of move strings regardless of the key format."""
        if "Moves" in example:  # For puzzles and full games
            moves = example["Moves"]
            if isinstance(moves, str):
                return moves.split()
            return moves
        elif "line" in example:  # For analysis positions
            line = example["line"]
            if isinstance(line, str):
                return line.split()
            return line
        return []

    def apply_moves_to_board(self, fen: str, moves: List[str], skip_count: int) -> str:
        """Apply `skip_count` moves to the board starting from `fen` and return the resulting fen."""
        board = chess.Board(fen)
        for move_str in moves[:skip_count]:
            move = chess.Move.from_uci(move_str)
            if move not in board.legal_moves:
                # If the move is somehow illegal given the fen, break or handle error
                # For safety, just break, but ideally your data should be consistent.
                break
            board.push(move)
        return board.fen()

    def debug_print_batch(self, examples: List[Dict[str, Any]], moves_list: List[List[str]]):
        """Print debug information about the length of each sample in the batch."""
        print("\nDebug: Sample Lengths")
        for ex, moves in zip(examples, moves_list):
            ex_type = ex.get("example_type", "unknown")
            if ex_type in ["full_game", "puzzle"]:
                print(f"Type: {ex_type}, Moves count: {len(moves)}")
            else:
                # For analysis or other types, just print a placeholder.
                print(f"Type: {ex_type}, (not applicable)")

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        if not examples:
            raise ValueError("Received empty examples list")

        input_ids_list = []
        attention_mask_list = []
        decoder_input_ids_list = []
        decoder_attention_mask_list = []
        labels_list = []
        regression_labels_list = []
        regression_mask_list = []

        all_subsequent_moves = []  # For debug

        for ex in examples:
            ex_type = ex.get("example_type")
            # Default FEN if none provided
            original_fen = ex.get("fen", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
            moves = self.get_moves(ex)

            if ex_type == "full_game":
                p = torch.rand(1).item()
                if len(moves) == 0:
                    # No moves
                    subsequent_moves = []
                    fen_used = original_fen
                else:
                    if p >= 0.7:
                        # 70% of the time: use full game from start
                        # Possibly truncated to max_moves
                        subsequent_moves = moves[:self.max_moves] if len(moves) > self.max_moves else moves
                        fen_used = original_fen
                    else:
                        # 30% of the time: randomly skip some initial portion of the game
                        # skip_count from [1, len(moves)-1] if you want to ensure skipping at least one move
                        # or [0, len(moves)] if you allow skipping zero moves occasionally.
                        # Let's say skip_count >=1 for effect:
                        skip_count = torch.randint(low=1, high=len(moves), size=()).item()
                        fen_used = self.apply_moves_to_board(original_fen, moves, skip_count)
                        subsequent_moves = moves[skip_count:skip_count+self.max_moves]
                move_ids = self.tokenizer.encode_moves(subsequent_moves)
                regression_val = None

            elif ex_type == "puzzle":
                # Just use all puzzle moves or truncated at max_moves
                subsequent_moves = moves[:self.max_moves] if len(moves) > self.max_moves else moves
                fen_used = original_fen
                move_ids = self.tokenizer.encode_moves(subsequent_moves)

                # Check if final position is checkmate
                board = chess.Board(fen_used)
                for m in subsequent_moves:
                    board.push(chess.Move.from_uci(m))
                ended_in_checkmate = board.is_checkmate()

                regression_val = None

                # Create decoder inputs & labels
                if len(move_ids) > 0:
                    decoder_input_ids = [self.tokenizer.bos_token_id] + move_ids
                    labels = move_ids.copy()

                    if ended_in_checkmate:
                        # If final position is checkmate, add EOS token
                        decoder_input_ids.append(self.tokenizer.eos_token_id)
                        labels.append(self.tokenizer.eos_token_id)
                else:
                    # If no moves at all, just start and end token if checkmate, otherwise just bos?
                    decoder_input_ids = [self.tokenizer.bos_token_id]
                    labels = []
                    if ended_in_checkmate:
                        decoder_input_ids.append(self.tokenizer.eos_token_id)
                        labels.append(self.tokenizer.eos_token_id)

            elif ex_type == "analysis":
                subsequent_moves = moves
                fen_used = original_fen
                move_ids = self.tokenizer.encode_moves(moves)
                cp = ex.get("cp", None)
                mate = ex.get("mate", None)
                if self.use_regression:
                    if mate is not None:
                        # Large positive/negative value for mate
                        regression_val = 40000 if mate > 0 else -40000
                    else:
                        regression_val = cp if cp is not None else 0
                else:
                    regression_val = None
            else:
                raise ValueError(
                    f"Unknown example_type. Expected 'full_game', 'puzzle', or 'analysis', got '{ex_type}'")

            # Determine regression mask
            is_analysis = (ex_type == "analysis") and (regression_val is not None)
            regression_labels_list.append(regression_val if regression_val is not None else 0.0)
            regression_mask_list.append(1.0 if is_analysis else 0.0)

            # Encoder input (FEN) after adjustments
            fen_ids = self.tokenizer.encode_fen(fen_used)
            encoder_input_ids = [self.tokenizer.bos_token_id] + fen_ids + [self.tokenizer.eos_token_id]
            encoder_attention_mask = [1]*len(encoder_input_ids)

            # Decoder input and labels
            if ex_type == "analysis":
                # Minimal decoder input for analysis
                decoder_input_ids = [self.tokenizer.bos_token_id]
                labels = [-100]
            else:
                # For puzzle/full_game
                if len(move_ids) > 0:
                    decoder_input_ids = [self.tokenizer.bos_token_id] + move_ids + [self.tokenizer.eos_token_id]
                    labels = move_ids + [self.tokenizer.eos_token_id]
                else:
                    # No moves
                    decoder_input_ids = [self.tokenizer.bos_token_id, self.tokenizer.eos_token_id]
                    labels = [self.tokenizer.eos_token_id]

            # Append to batch lists
            input_ids_list.append(encoder_input_ids)
            attention_mask_list.append(encoder_attention_mask)
            decoder_input_ids_list.append(decoder_input_ids)
            decoder_attention_mask_list.append([1]*len(decoder_input_ids))
            labels_list.append(labels)
            all_subsequent_moves.append(subsequent_moves)  # For debug

        # Padding function
        def pad_seq(seqs, max_len, pad_id):
            return [seq + [pad_id]*(max_len - len(seq)) for seq in seqs]

        max_enc_len = max(len(seq) for seq in input_ids_list)
        max_dec_len = max(len(seq) for seq in decoder_input_ids_list)

        padded_input_ids = pad_seq(input_ids_list, max_enc_len, self.tokenizer.pad_token_id)
        padded_attention_mask = pad_seq(attention_mask_list, max_enc_len, 0)
        padded_decoder_input_ids = pad_seq(decoder_input_ids_list, max_dec_len, self.tokenizer.pad_token_id)
        padded_decoder_attention_mask = pad_seq(decoder_attention_mask_list, max_dec_len, 0)
        padded_labels = pad_seq(labels_list, max_dec_len, -100)

        regression_targets = [float(x) for x in regression_labels_list]

        batch = {
            "fen_input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "fen_attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
            "decoder_input_ids": torch.tensor(padded_decoder_input_ids, dtype=torch.long),
            "decoder_attention_mask": torch.tensor(padded_decoder_attention_mask, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }

        if self.use_regression:
            batch["regression_labels"] = torch.tensor(regression_targets, dtype=torch.float)
            batch["regression_mask"] = torch.tensor(regression_mask_list, dtype=torch.float)

        # Debug print
        if self.debug:
            self.debug_print_batch(examples, all_subsequent_moves)

        return batch
