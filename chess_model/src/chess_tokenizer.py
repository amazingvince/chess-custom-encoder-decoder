from tokenizers import Tokenizer, pre_tokenizers, decoders, processors
from tokenizers.models import WordPiece
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple, Union, Optional
import chess
import chess.pgn
from io import StringIO
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils_base import AddedToken
import json
import os
import torch

@dataclass
class ChessVocabSpec:
    """Specification for chess vocabulary with dual representation"""
    
    # Special tokens
    SPECIAL_TOKENS = {
        "<PAD>": 0,
        "<UNK>": 1,
        "<CLS>": 2,
        "<SEP>": 3,
        "<MASK>": 4,
        "<START>": 5,
        "<EOS>": 6,
        "<EP-AVAILABLE>": 7,  # Single token for en passant flag
    }
    
    # Piece tokens with dual representation
    PIECE_TOKENS = {
        # White pieces
        "P": {"symbol": "♙", "id": 8},
        "N": {"symbol": "♘", "id": 9},
        "B": {"symbol": "♗", "id": 10},
        "R": {"symbol": "♖", "id": 11},
        "Q": {"symbol": "♕", "id": 12},
        "K": {"symbol": "♔", "id": 13},
        # Black pieces
        "p": {"symbol": "♟", "id": 14},
        "n": {"symbol": "♞", "id": 15},
        "b": {"symbol": "♝", "id": 16},
        "r": {"symbol": "♜", "id": 17},
        "q": {"symbol": "♛", "id": 18},
        "k": {"symbol": "♚", "id": 19},
        # Empty square
        ".": {"symbol": "·", "id": 20}
    }
    
    # State tokens
    STATE_TOKENS = {
        "<W-CASTLE-K>": 21,
        "<W-CASTLE-Q>": 22,
        "<B-CASTLE-K>": 23,
        "<B-CASTLE-Q>": 24,
        "<WHITE-MOVE>": 25,
        "<BLACK-MOVE>": 26,
    }
    
    # Square tokens for moves and board positions (a1-h8)
    SQUARE_TOKENS = {
        chess.square_name(square): 27 + square
        for square in chess.SQUARES
    }
    
    # Promotion tokens
    PROMOTION_TOKENS = {
        "<PROMOTE-Q>": 91,
        "<PROMOTE-R>": 92,
        "<PROMOTE-B>": 93,
        "<PROMOTE-N>": 94,
    }
    
    # Mapping from piece type to promotion token
    PROMOTION_MAP = {
        chess.QUEEN: "<PROMOTE-Q>",
        chess.ROOK: "<PROMOTE-R>",
        chess.BISHOP: "<PROMOTE-B>",
        chess.KNIGHT: "<PROMOTE-N>",
    }

    def get_vocab(self) -> Dict[str, int]:
        """Returns complete vocabulary mapping"""
        vocab = self.SPECIAL_TOKENS.copy()
        
        # Add pieces with both representations
        for piece, info in self.PIECE_TOKENS.items():
            vocab[piece] = info["id"]
            vocab[info["symbol"]] = info["id"]
            
        vocab.update(self.STATE_TOKENS)
        vocab.update(self.SQUARE_TOKENS)
        vocab.update(self.PROMOTION_TOKENS)
        return vocab
    
    def get_decoder_map(self) -> Dict[int, str]:
        """Returns mapping from token IDs to preferred decoded form"""
        decoder_map = {v: k for k, v in self.SPECIAL_TOKENS.items()}
        
        # Pieces decode to symbols
        for piece, info in self.PIECE_TOKENS.items():
            decoder_map[info["id"]] = info["symbol"]
            
        decoder_map.update({v: k for k, v in self.STATE_TOKENS.items()})
        decoder_map.update({v: k for k, v in self.SQUARE_TOKENS.items()})
        decoder_map.update({v: k for k, v in self.PROMOTION_TOKENS.items()})
        return decoder_map





class ChessTokenizer(PreTrainedTokenizer):
    """Chess tokenizer compatible with HuggingFace transformers library.
    
    This tokenizer handles both chess positions (in FEN notation) and moves (in UCI format).
    It supports both single position encoding and batch processing, with features for:
    - Position encoding from FEN strings
    - Move sequence encoding
    - Special token handling (CLS, SEP, MASK, etc.)
    - Padding and truncation
    - Both string and tensor outputs
    """
    
    vocab_files_names = {"vocab_file": "chess_vocab.json"}
    model_input_names = ["input_ids", "attention_mask"]
    
    def __init__(
        self,
        vocab_file=None,
        unk_token="<UNK>",
        sep_token="<SEP>",
        pad_token="<PAD>",
        cls_token="<CLS>",
        mask_token="<MASK>",
        eos_token="<EOS>",
        **kwargs
    ):
        # Initialize vocabulary specification
        self.vocab_spec = ChessVocabSpec()
        
        # Initialize vocabulary before calling super().__init__
        if vocab_file is not None and os.path.exists(vocab_file):
            with open(vocab_file, 'r', encoding='utf-8') as f:
                self.vocab = json.load(f)
        else:
            self.vocab = self.vocab_spec.get_vocab()
            
        self.decoder_map = self.vocab_spec.get_decoder_map()
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
        
        # Now call super().__init__
        super().__init__(
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            eos_token=eos_token,
            **kwargs
        )
        
    @property
    def vocab_size(self) -> int:
        return len(self.vocab)
        
    def get_vocab(self) -> Dict[str, int]:
        return self.vocab.copy()
        
    def _tokenize(self, text: str, add_eos: bool = True) -> List[str]:
        """Tokenize text into subtokens"""
        if not text.strip():
            return ["<CLS>", "<SEP>"]
            
        if text.strip().count('/') == 7:  # Looks like FEN
            try:
                board = chess.Board(text)
                return ["<CLS>"] + self._board_to_tokens(board) + ["<SEP>"]
            except ValueError as e:
                raise ValueError(f"Invalid FEN string: {text}") from e
        else:  # Assume moves
            return self._tokenize_moves(text, add_eos=add_eos)
            
    def _tokenize_moves(self, moves_str: str, add_eos: bool = True) -> List[str]:
        """Tokenize chess moves"""
        if not moves_str.strip():
            return ["<START>"]
            
        tokens = ["<START>"]
        board = chess.Board()
        
        moves = moves_str.split()
        for i, move_str in enumerate(moves):
            try:
                move = chess.Move.from_uci(move_str)
                if move not in board.legal_moves:
                    tokens.append(self.unk_token)
                    continue
                    
                from_square = chess.square_name(move.from_square)
                to_square = chess.square_name(move.to_square)
                tokens.extend([from_square, to_square])
                
                if move.promotion is not None:
                    promotion_token = self.vocab_spec.PROMOTION_MAP[move.promotion]
                    tokens.append(promotion_token)
                    
                board.push(move)
                
                is_last_move = i == len(moves) - 1
                if is_last_move and (add_eos or board.is_game_over()):
                    tokens.append("<EOS>")
                    
            except (ValueError, AssertionError):
                tokens.append(self.unk_token)
                
        return tokens
        
    def _convert_token_to_id(self, token: str) -> int:
        """Convert token to ID"""
        return self.vocab.get(token, self.vocab[self.unk_token])
        
    def _convert_id_to_token(self, index: int) -> str:
        """Convert ID to token"""
        return self.ids_to_tokens.get(index, self.unk_token)
        
    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Convert tokens to string"""
        return " ".join(tokens)
        
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """Build model inputs from a sequence by adding special tokens."""
        if token_ids_1 is None:
            return token_ids_0
        return token_ids_0 + [self.sep_token_id] + token_ids_1
        
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """Save the tokenizer vocabulary to a file"""
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory)
            
        vocab_file = os.path.join(
            save_directory, 
            (filename_prefix + "-" if filename_prefix else "") + "chess_vocab.json"
        )
            
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
            
        return (vocab_file,)
        
    def _board_to_tokens(self, board: chess.Board) -> List[str]:
        """Convert chess.Board to list of tokens"""
        tokens = []
        
        # Encode each square
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None:
                tokens.append(".")
            else:
                tokens.append(piece.symbol())
                
        # Encode game state
        tokens.append("<WHITE-MOVE>" if board.turn else "<BLACK-MOVE>")
        
        # Castling rights
        if board.has_kingside_castling_rights(chess.WHITE):
            tokens.append("<W-CASTLE-K>")
        if board.has_queenside_castling_rights(chess.WHITE):
            tokens.append("<W-CASTLE-Q>")
        if board.has_kingside_castling_rights(chess.BLACK):
            tokens.append("<B-CASTLE-K>")
        if board.has_queenside_castling_rights(chess.BLACK):
            tokens.append("<B-CASTLE-Q>")
            
        # En passant square (if available)
        if board.ep_square is not None:
            tokens.append("<EP-AVAILABLE>")
            tokens.append(chess.square_name(board.ep_square))
            
        return tokens

    def encode_fen(
        self,
        fen: str,
        padding: bool = False,
        max_length: Optional[int] = None,
        truncation: bool = True,
        return_tensors: Optional[str] = None,
    ) -> Dict[str, Union[List[int], 'torch.Tensor']]:
        """
        Encode a FEN string with special tokens (CLS and SEP).
        
        Args:
            fen: FEN string representing chess position
            padding: Whether to pad sequences
            max_length: Maximum sequence length for padding
            truncation: Whether to truncate sequences longer than max_length
            return_tensors: Output format ("pt" for PyTorch tensors)
            
        Returns:
            Dictionary with input_ids and attention_mask
        """
        try:
            # Validate FEN
            board = chess.Board(fen)
            
            # Get tokens with special tokens
            tokens = ["<CLS>"] + self._board_to_tokens(board) + ["<SEP>"]
            
            # Convert to ids
            input_ids = [self._convert_token_to_id(t) for t in tokens]
            attention_mask = [1] * len(input_ids)
            
            # Handle truncation
            if truncation and max_length is not None and len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
                attention_mask = attention_mask[:max_length]
            
            # Handle padding
            if padding and max_length is not None:
                pad_length = max_length - len(input_ids)
                if pad_length > 0:
                    input_ids = input_ids + [self.pad_token_id] * pad_length
                    attention_mask = attention_mask + [0] * pad_length
            
            # Prepare outputs
            encoded = {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }
            
            # Convert to tensors if requested
            if return_tensors == "pt":
                import torch
                encoded = {
                    "input_ids": torch.tensor([input_ids]),
                    "attention_mask": torch.tensor([attention_mask])
                }
                
            return encoded
            
        except ValueError as e:
            raise ValueError(f"Invalid FEN string: {fen}") from e

    def encode_game_at_position(
        self, 
        moves: Union[str, List[str]], 
        position_idx: int, 
        add_special_tokens: bool = True,
        add_eos: bool = True,
        padding: bool = False,
        max_length: Optional[int] = None,
        truncation: bool = True,
        return_tensors: Optional[str] = None,
        starting_fen: Optional[str] = None
    ) -> Dict[str, Union[List[int], 'torch.Tensor']]:
        """
        Encode game state at specific position and remaining moves.
        
        Args:
            moves: String or list of moves in UCI format (e.g., "e2e4 e7e5" or ["e2e4", "e7e5"]),
                or PGN format (e.g., "1. e4 e5 2. Nf3")
            position_idx: Index of position to start from (0-based)
            add_special_tokens: Whether to add CLS/SEP tokens
            add_eos: Whether to add EOS token at end of sequence
            padding: Whether to pad sequence
            max_length: Maximum sequence length
            truncation: Whether to truncate sequence
            return_tensors: Output format ("pt" for PyTorch tensors)
            starting_fen: Optional starting position in FEN format
            
        Returns:
            Dictionary with input_ids and attention_mask
        """
        # Initialize board with starting position
        board = chess.Board(starting_fen) if starting_fen else chess.Board()
        parsed_moves = []
        
        # Parse and validate moves based on input type
        try:
            if isinstance(moves, list):
                # Handle list of UCI moves
                for move_str in moves:
                    move = chess.Move.from_uci(move_str)
                    if move not in board.legal_moves:
                        raise ValueError(f"Illegal move: {move_str} in position {board.fen()}")
                    parsed_moves.append(move)
                    board.push(move)
            else:
                moves_str = str(moves).strip()
                # Handle PGN format
                if any(c.isdigit() and c + '.' in moves_str for c in moves_str):
                    game = chess.pgn.read_game(StringIO(moves_str))
                    if game is None:
                        raise ValueError("Invalid PGN format")
                    parsed_moves = list(game.mainline_moves())
                else:
                    # Handle space-separated UCI moves
                    for move_str in moves_str.split():
                        move = chess.Move.from_uci(move_str)
                        if move not in board.legal_moves:
                            raise ValueError(f"Illegal move: {move_str} in position {board.fen()}")
                        parsed_moves.append(move)
                        board.push(move)
        except ValueError as e:
            raise ValueError(f"Error parsing moves: {str(e)}") from e
            
        # Validate position_idx
        if position_idx < 0 or position_idx > len(parsed_moves):
            raise ValueError(f"Invalid position_idx: {position_idx}. Must be between 0 and {len(parsed_moves)}")
            
        # Reset board and play moves until position_idx
        board = chess.Board(starting_fen) if starting_fen else chess.Board()
        for move in parsed_moves[:position_idx]:
            board.push(move)
            
        # Get position tokens
        pos_tokens = self._board_to_tokens(board)
        if add_special_tokens:
            pos_tokens = ["<CLS>"] + pos_tokens + ["<SEP>"]
            
        # Get remaining moves tokens
        remaining_moves = parsed_moves[position_idx:]
        move_tokens = ["<START>"]
        
        current_board = board.copy()
        for i, move in enumerate(remaining_moves):
            from_square = chess.square_name(move.from_square)
            to_square = chess.square_name(move.to_square)
            move_tokens.extend([from_square, to_square])
            
            if move.promotion is not None:
                promotion_token = self.vocab_spec.PROMOTION_MAP[move.promotion]
                move_tokens.append(promotion_token)
                
            current_board.push(move)
            
            # Add EOS token if:
            # 1. It's the last move AND (add_eos is True OR game is over)
            # 2. Game is over (checkmate, stalemate, etc.)
            is_last_move = i == len(remaining_moves) - 1
            if current_board.is_game_over() or (is_last_move and add_eos):
                move_tokens.append("<EOS>")
                
        # Convert tokens to ids
        tokens = pos_tokens + move_tokens
        input_ids = [self._convert_token_to_id(t) for t in tokens]
        attention_mask = [1] * len(input_ids)
        
        # Handle truncation
        if truncation and max_length is not None and len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]
            
        # Handle padding
        if padding and max_length is not None:
            pad_length = max_length - len(input_ids)
            if pad_length > 0:
                input_ids = input_ids + [self.pad_token_id] * pad_length
                attention_mask = attention_mask + [0] * pad_length
                
        # Prepare outputs
        encoded = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
        
        # Convert to tensors if requested
        if return_tensors == "pt":
            import torch
            encoded = {
                "input_ids": torch.tensor([input_ids]),
                "attention_mask": torch.tensor([attention_mask])
            }
            
        return encoded

    def encode_position(
        self,
        fen: Optional[str] = None,
        moves: Optional[Union[str, List[str]]] = None,
        return_tensors: Optional[str] = "pt",
        padding: bool = True,
        max_length: Optional[int] = 256,
        truncation: bool = True,
        add_special_tokens: bool = True,
        add_eos: bool = True  # Add this parameter
    ) -> Dict[str, Union[List[int], 'torch.Tensor']]:
        """
        Encode chess position and optional moves.
        
        Args:
            fen: FEN string or None (uses starting position)
            moves: Moves in UCI format, PGN format, or list format
            return_tensors: Output format ("pt" for PyTorch tensors)
            padding: Whether to pad sequences
            max_length: Maximum sequence length
            truncation: Whether to truncate sequences
            add_special_tokens: Whether to add CLS/SEP tokens
            add_eos: Whether to add EOS token at the end of move sequences
            
        Returns:
            Dictionary with input_ids and attention_mask
        """
        # Input validation
        if max_length is not None and max_length <= 0:
            raise ValueError("max_length must be positive")
            
        # Use starting position if no FEN provided
        if fen is None:
            fen = chess.STARTING_FEN
            
        try:
            chess.Board(fen)  # Validate FEN
        except ValueError as e:
            raise ValueError(f"Invalid FEN string: {fen}") from e
            
        # If no moves, just encode the FEN
        if moves is None:
            return self.encode_fen(
                fen=fen,
                padding=padding,
                max_length=max_length,
                truncation=truncation,
                return_tensors=return_tensors
            )
            
        # Parse moves to UCI format
        try:
            uci_moves = self._parse_moves(moves)
        except ValueError as e:
            raise ValueError(f"Error parsing moves: {moves}") from e
            
        # Validate moves and position
        board = chess.Board(fen)
        for move in uci_moves:
            try:
                chess_move = chess.Move.from_uci(move)
                if chess_move not in board.legal_moves:
                    raise ValueError(f"Illegal move: {move} in position {board.fen()}")
                board.push(chess_move)
            except ValueError as e:
                raise ValueError(f"Invalid move format: {move}") from e
                
        return self.encode_game_at_position(
            " ".join(uci_moves),
            position_idx=0,
            add_special_tokens=add_special_tokens,
            add_eos=add_eos,  
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            return_tensors=return_tensors
    )

    def _parse_moves(self, moves: Optional[Union[str, List[str]]] = None) -> List[str]:
        """
        Parse moves in various formats to UCI format.
        
        Args:
            moves: Moves in UCI format, PGN format, or list of moves
            
        Returns:
            List of moves in UCI format
        """
        if moves is None:
            return []
            
        # Already a list of UCI moves
        if isinstance(moves, list):
            return moves
            
        # String input
        moves_str = moves.strip()
        if not moves_str:
            return []
            
        # Check if PGN format (contains move numbers like "1.")
        if any(c.isdigit() and c + '.' in moves_str for c in moves_str):
            try:
                game = chess.pgn.read_game(StringIO(moves_str))
                if game is None:
                    raise ValueError("Invalid PGN format")
                return [move.uci() for move in game.mainline_moves()]
            except Exception as e:
                raise ValueError(f"Error parsing PGN moves: {moves_str}") from e
                
        # Assume space-separated UCI moves
        return moves_str.split()


    def batch_encode_positions(
        self,
        fens: Optional[List[Optional[str]]] = None,
        moves: Optional[List[Optional[Union[str, List[str]]]]] = None,
        return_tensors: Optional[str] = "pt",
        padding: bool = True,
        max_length: Optional[int] = 256,
        truncation: bool = True,
        add_special_tokens: bool = True,
        add_eos: bool = True  # Add this parameter
    ) -> Union[Dict[str, 'torch.Tensor'], List[Dict[str, Union[List[int], 'torch.Tensor']]]]:

        """
        Encode a batch of chess positions and moves.
        
        Args:
            fens: List of FEN strings or None (uses starting position)
            moves: List of move strings or lists (UCI, PGN, or list format)
            return_tensors: Output format ("pt" for PyTorch tensors)
            padding: Whether to pad sequences
            max_length: Maximum sequence length
            truncation: Whether to truncate sequences
            add_special_tokens: Whether to add CLS/SEP tokens
            
        Returns:
            Dictionary with batched input_ids and attention_mask tensors,
            or list of dictionaries if return_tensors is None
        """
        # Handle None inputs
        batch_size = max(
            len(fens) if fens is not None else 0,
            len(moves) if moves is not None else 0
        )
        if batch_size == 0:
            raise ValueError("At least one of fens or moves must be provided")
            
        if fens is None:
            fens = [None] * batch_size
        if moves is None:
            moves = [None] * batch_size
            
        # Ensure equal lengths
        if len(fens) != len(moves):
            raise ValueError("Number of FENs must match number of moves")
            
        # Process each position in batch
        encodings = []
        for fen, move in zip(fens, moves):
            try:
                encoding = self.encode_position(
                    fen=fen,
                    moves=move,
                    return_tensors=None,  # Handle tensors after batch processing
                    padding=padding,
                    max_length=max_length,
                    truncation=truncation,
                    add_special_tokens=add_special_tokens,
                    add_eos=add_eos  # Pass add_eos parameter
                )
                encodings.append(encoding)
            except Exception as e:
                raise ValueError(f"Error processing FEN: {fen}, moves: {move}") from e
                
        # Handle padding and tensor conversion as before
        if padding:
            actual_max_len = max(len(enc["input_ids"]) for enc in encodings)
            if max_length is None:
                max_length = actual_max_len
            else:
                max_length = min(max_length, actual_max_len)
            
            # Pad all sequences
            for enc in encodings:
                pad_length = max_length - len(enc["input_ids"])
                if pad_length > 0:
                    enc["input_ids"].extend([self.pad_token_id] * pad_length)
                    enc["attention_mask"].extend([0] * pad_length)
                    
        # Convert to tensors if requested
        if return_tensors == "pt":
            import torch
            batch_encodings = {
                "input_ids": torch.tensor([enc["input_ids"] for enc in encodings]),
                "attention_mask": torch.tensor([enc["attention_mask"] for enc in encodings])
            }
            return batch_encodings
            
        return encodings



    def __call__(
        self,
        fens: Optional[Union[str, List[Optional[str]]]] = None,
        moves: Optional[Union[str, List[Optional[Union[str, List[str]]]]]] = None,
        add_eos: bool = True,  # Add this parameter
        **kwargs
    ) -> Union[Dict[str, Union[List[int], 'torch.Tensor']], List[Dict[str, Union[List[int], 'torch.Tensor']]]]:
        """
        Main entry point for tokenization. 
        Handles both single positions and batches.
        
        Args:
            fens: Single FEN string or list of FENs
            moves: Single moves string/list or list of moves
            **kwargs: Additional arguments passed to encode_position/batch_encode_positions
            
        Returns:
            Encoded position(s) with input_ids and attention_mask
        """
        # Handle batch inputs
        if isinstance(fens, list) or (
            isinstance(moves, list) and any(isinstance(m, list) for m in moves if m is not None)
        ):
            return self.batch_encode_positions(fens=fens, moves=moves, add_eos=add_eos, **kwargs)
        
        # Handle single inputs
        return self.encode_position(fen=fens, moves=moves, add_eos=add_eos, **kwargs)

    
    def decode(
            self, 
            token_ids: Union[int, List[int], 'torch.Tensor'], 
            skip_special_tokens: bool = False,
            **kwargs
        ) -> str:
            """Decode token IDs to string."""
            if isinstance(token_ids, torch.Tensor):
                if token_ids.dim() > 1:
                    token_ids = token_ids.squeeze(0)
                token_ids = token_ids.tolist()
                
            tokens = []
            for idx in token_ids:
                token = self._convert_id_to_token(idx)
                if skip_special_tokens and token.startswith('<'):
                    continue
                tokens.append(token)
                
            return ' '.join(tokens)
    
    def _tokenize_moves(self, moves_str: str, add_eos: bool = True) -> List[str]:
        """Tokenize chess moves"""
        if not moves_str.strip():
            return ["<START>"]
            
        tokens = ["<START>"]
        board = chess.Board()
        
        # Handle PGN format
        if any(c.isdigit() and c + '.' in moves_str for c in moves_str):
            try:
                game = chess.pgn.read_game(StringIO(moves_str))
                if game is None:
                    raise ValueError("Invalid PGN format")
                moves = [move.uci() for move in game.mainline_moves()]
                moves_str = " ".join(moves)
            except Exception as e:
                raise ValueError(f"Error parsing PGN moves: {moves_str}") from e
        
        moves = moves_str.split()
        for i, move_str in enumerate(moves):
            try:
                move = chess.Move.from_uci(move_str)
                if move not in board.legal_moves:
                    tokens.append(self.unk_token)
                    continue
                    
                from_square = chess.square_name(move.from_square)
                to_square = chess.square_name(move.to_square)
                tokens.extend([from_square, to_square])
                
                if move.promotion is not None:
                    promotion_token = self.vocab_spec.PROMOTION_MAP[move.promotion]
                    tokens.append(promotion_token)
                    
                board.push(move)
                
                # Add EOS token if:
                # 1. It's the last move AND (add_eos is True OR game is over)
                # 2. Game is over (checkmate, stalemate, etc.)
                is_last_move = i == len(moves) - 1
                if board.is_game_over() or (is_last_move and add_eos):
                    tokens.append("<EOS>")
                    
            except (ValueError, AssertionError):
                tokens.append(self.unk_token)
                
        return tokens