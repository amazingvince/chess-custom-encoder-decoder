import torch
from dataclasses import dataclass
from typing import List, Dict, Any

from chess_tokenizer import ChessTokenizer

@dataclass
class MixedDataCollator:
    tokenizer: "ChessTokenizer"
    max_moves: int = 50
    use_regression: bool = True

    def get_moves(self, example: Dict[str, Any]) -> List[str]:
        """Helper method to get moves regardless of key case"""
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

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        if not examples:
            raise ValueError("Received empty examples list")
        
        batch = super().__call__(examples)
        
        for i, ex in enumerate(examples):
            if not ex:
                raise ValueError(f"Example at index {i} is empty: {ex}")

        input_ids_list = []
        attention_mask_list = []
        decoder_input_ids_list = []
        decoder_attention_mask_list = []
        labels_list = []
        regression_labels_list = []
        regression_mask_list = []

        for ex in examples:
            ex_type = ex.get("example_type")

            # Default FEN if none provided
            fen = ex.get("fen", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
            fen_ids = self.tokenizer.encode_fen(fen)

            if ex_type == "full_game":
                moves = self.get_moves(ex)
                if len(moves) > 0:
                    start_idx = torch.randint(low=0, high=len(moves), size=()).item()
                    subsequent_moves = moves[start_idx:start_idx+self.max_moves]
                else:
                    subsequent_moves = []
                move_ids = self.tokenizer.encode_moves(subsequent_moves)
                regression_val = None

            elif ex_type == "puzzle":
                moves = self.get_moves(ex)
                move_ids = self.tokenizer.encode_moves(moves)
                regression_val = None

            elif ex_type == "analysis":
                moves = self.get_moves(ex)
                move_ids = self.tokenizer.encode_moves(moves)
                cp = ex.get("cp", None)
                mate = ex.get("mate", None)
                if self.use_regression:
                    if mate is not None:
                        # Large positive/negative value for mate
                        regression_val = 100 if mate > 0 else -100
                    else:
                        regression_val = cp if cp is not None else 0
                else:
                    regression_val = None
            else:
                raise ValueError(f"Unknown example_type. Expected 'full_game', 'puzzle', or 'analysis', but got '{ex_type}'. Full example: {ex}")

            # Only analysis samples should contribute to regression. Others will have None and be masked out.
            regression_labels_list.append(regression_val)
            regression_mask_list.append(1.0 if (ex_type == "analysis" and regression_val is not None) else 0.0)

            # Encoder input: fen
            encoder_input_ids = [self.tokenizer.bos_token_id] + fen_ids + [self.tokenizer.eos_token_id]
            encoder_attention_mask = [1]*len(encoder_input_ids)

            # Decoder input: moves
            if len(move_ids) > 0:
                decoder_input_ids = [self.tokenizer.bos_token_id] + move_ids + [self.tokenizer.eos_token_id]
                labels = move_ids + [self.tokenizer.eos_token_id]
            else:
                decoder_input_ids = [self.tokenizer.bos_token_id, self.tokenizer.eos_token_id]
                labels = [self.tokenizer.eos_token_id]

            input_ids_list.append(encoder_input_ids)
            attention_mask_list.append(encoder_attention_mask)
            decoder_input_ids_list.append(decoder_input_ids)
            decoder_attention_mask_list.append([1]*len(decoder_input_ids))
            labels_list.append(labels)

        def pad_seq(seqs, max_len, pad_id):
            return [seq + [pad_id]*(max_len - len(seq)) for seq in seqs]

        max_enc_len = max(len(seq) for seq in input_ids_list)
        max_dec_len = max(len(seq) for seq in decoder_input_ids_list)

        padded_input_ids = pad_seq(input_ids_list, max_enc_len, self.tokenizer.pad_token_id)
        padded_attention_mask = pad_seq(attention_mask_list, max_enc_len, 0)
        padded_decoder_input_ids = pad_seq(decoder_input_ids_list, max_dec_len, self.tokenizer.pad_token_id)
        padded_decoder_attention_mask = pad_seq(decoder_attention_mask_list, max_dec_len, 0)
        padded_labels = pad_seq(labels_list, max_dec_len, -100)

        regression_targets = [0.0 if x is None else float(x) for x in regression_labels_list]

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

        return batch
