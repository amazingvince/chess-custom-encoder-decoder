import pytest
import torch
from typing import List, Dict, Union
import chess

from chess_model.src.chess_tokenizer import ChessTokenizer

def test_positions() -> List[Dict[str, Union[str, List[str]]]]:
    """Return test positions and moves for consistent testing"""
    return [
        {
            "fen": chess.STARTING_FEN,
            "moves": "e2e4 e7e6",
            "pgn": "1. e4 e6",
            "moves_list": ["e2e4", "e7e6"]
        },
    ]

@pytest.fixture
def tokenizer():
    return ChessTokenizer()

def test_init(tokenizer):
    """Test tokenizer initialization"""
    assert tokenizer.vocab_size > 0
    assert "<PAD>" in tokenizer.vocab
    assert "<CLS>" in tokenizer.vocab
    assert "<SEP>" in tokenizer.vocab
    assert "<MASK>" in tokenizer.vocab
    assert "<UNK>" in tokenizer.vocab

def test_encode_single_fen(tokenizer):
    """Test encoding a single FEN string"""
    test_data = test_positions()[0]
    encoded = tokenizer(fens=test_data["fen"], max_length=100, padding=False)
    
    assert isinstance(encoded, dict)
    assert "input_ids" in encoded
    assert "attention_mask" in encoded
    assert isinstance(encoded["input_ids"], torch.Tensor)
    assert encoded["input_ids"].dim() == 2  # [batch_size=1, sequence_length]
    assert tokenizer.decode(encoded["input_ids"]) == "<CLS> ♖ ♘ ♗ ♕ ♔ ♗ ♘ ♖ ♙ ♙ ♙ ♙ ♙ ♙ ♙ ♙ · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · ♟ ♟ ♟ ♟ ♟ ♟ ♟ ♟ ♜ ♞ ♝ ♛ ♚ ♝ ♞ ♜ <WHITE-MOVE> <W-CASTLE-K> <W-CASTLE-Q> <B-CASTLE-K> <B-CASTLE-Q> <SEP>"


def test_encode_single_moves(tokenizer):
    """Test encoding moves in different formats"""
    test_data = test_positions()[0]
    
    # Test UCI string format
    encoded_uci = tokenizer(moves=test_data["moves"])
    
    # Test moves list format
    encoded_list = tokenizer(moves=test_data["moves_list"])
    
    # Test PGN format
    encoded_pgn = tokenizer(moves=test_data["pgn"])
    
    # All formats should produce the same encoding
    assert torch.equal(encoded_uci["input_ids"], encoded_list["input_ids"])
    assert torch.equal(encoded_uci["input_ids"], encoded_pgn["input_ids"])

def test_batch_encode(tokenizer):
    """Test batch encoding of positions and moves"""
    test_data = test_positions()
    
    # Prepare batch data
    fens = [test_data[0]["fen"]] * 5
    moves = [test_data[0]["moves"]] * 5
    
    # Test batch encoding
    encoded = tokenizer(fens=fens, moves=moves)
    
    assert isinstance(encoded, dict)
    assert "input_ids" in encoded
    assert "attention_mask" in encoded
    assert isinstance(encoded["input_ids"], torch.Tensor)
    assert encoded["input_ids"].dim() == 2  # [batch_size, sequence_length]
    assert encoded["input_ids"].size(0) == len(moves)

def test_none_handling(tokenizer):
    """Test handling of None values in batch encoding"""
    # Test None FEN (should use starting position)
    encoded_none_fen = tokenizer(fens=None, moves="e2e4 e7e6")
    encoded_start_fen = tokenizer(fens=chess.STARTING_FEN, moves="e2e4 e7e6")
    assert torch.equal(encoded_none_fen["input_ids"], encoded_start_fen["input_ids"])
    
    # Test None moves (should encode just the position)
    encoded_none_moves = tokenizer(fens=chess.STARTING_FEN, moves=None, padding=False)
    assert encoded_none_moves["input_ids"].size(1) < encoded_start_fen["input_ids"].size(1)


def test_invalid_fen(tokenizer):
    """Test handling of invalid FEN strings"""
    with pytest.raises(ValueError, match="Invalid FEN"):
        tokenizer(fens="invalid")



def test_attention_mask(tokenizer):
    """Test attention mask generation"""
    test_data = test_positions()
    
    # Encode batch with padding
    encoded = tokenizer(
        fens=[pos["fen"] for pos in test_data],
        moves=[pos["moves"] for pos in test_data],
        padding=True
    )
    
    # Check attention mask
    assert torch.all(encoded["attention_mask"] >= 0)
    assert torch.all(encoded["attention_mask"] <= 1)
    # Non-padded tokens should have attention mask 1
    assert torch.all(encoded["attention_mask"][:, 0] == 1)  # First token always attended
    # Padded tokens should have attention mask 0
    if encoded["attention_mask"].size(1) > 10:  # Assuming we have padding
        assert torch.any(encoded["attention_mask"] == 0)

def test_long_game(tokenizer):
    """Test encoding of a long chess game"""
    long_moves = [ "d2d4", "f7f5", "g2g3", "g7g6", "f1g2", "f8g7", "g1f3", "d7d6", "c2c3", "e7e6", "a2a4", "g8f6", "d1c2", "d8e7", "b1d2", "e6e5", "d4e5", "d6e5", "e2e4", "b8c6", "e1g1", "f5e4", "d2e4", "c8f5", "f3d2", "e8c8", "b2b4", "g7h6", "f1e1", "h6d2", "c1d2", "f6e4", "g2e4", "e7e6", "d2g5", "d8d6", "a1d1", "d6d1", "e1d1", "h7h6", "g5e3", "a7a5", "c2b1", "h6h5", "b4b5", "c6e7", "e3g5", "h8e8", "h2h4", "e6c4", "d1e1", "f5e4", "e1e4", "c4e6", "g5f4", "e6f5", "f4e5", "e7d5", "b1e1", "d5b6", "f2f4", "b6d7", "e1e2", "b7b6", "e4e3", "e8e7", "e3e4", "d7c5", "e4d4", "e7d7", "g1g2", "c8d8", "g2h2", "d8c8", "e2g2", "c8b8", "g2a2", "b8a7", "a2g2", "a7b8", "g2e2", "b8c8", "e2f3", "c8b8", "f3d1", "b8c8", "d1e2", "c8b8", "e2d1", "b8b7", "d4d7", "c5d7", "e5d4", "d7c5", "h2g2", "f5d5", "g2g1", "d5f5", "d4c5", "f5c5", "d1d4", "c5f5", "d4d2", "f5b1", "g1f2", "b1b3", "d2d4", "b3c2", "f2e3", "b7c8", "d4h8", "c8b7", "h8d4", "b7b8", "d4d8", "b8b7", "d8d5", "b7b8", "d5g8", "b8b7", "g8c4", "b7b8", "c4g8", "b8b7", "g8d5", "b7b8", "d5d8", "b8b7", "d8d4", "b7b8", "d4d8", "b8b7", "d8d3", "c2a4", "d3g6", "a4b5", "g6e4", "b7a7", "f4f5", "a5a4", "f5f6", "a4a3", "f6f7", "b5c5", "e3e2", "c5b5", "e2e3", "b5c5", "e3d3", "c5b5", "d3e3", "b5c5", "e3d3", "c5b5", "e4c4", "b5c4", "d3c4", "a3a2", "f7f8q", "a2a1q", "f8f3", "a1b1", "f3h5", "b1e4", "c4b3", "e4b1", "b3a3", "b1c1", "a3b3", "c1b1", "b3c4", "b1e4", "c4b3", "e4b1", "b3a4", "b1a2", "a4b4", "a2b1", "b4a4", "b1c2", "a4b4", "c2b1", "b4a4", "b1a2", "a4b4", "a2b1", "b4a4", "b1c2", "a4b4", "c2b1", "b4c4", "b1e4", "c4b3", "e4b1", "b3c4", "b1e4", "c4b3", "e4b1" ]  # 100 half-moves
    
    # Test with and without max_length
    encoded_no_limit = tokenizer(moves=long_moves, max_length=None)
    encoded_limited = tokenizer(moves=long_moves, max_length=200)
    
    assert encoded_limited["input_ids"].size(1) <=  encoded_no_limit["input_ids"].size(1)


def test_special_tokens_existence(tokenizer):
    """Test that all special tokens exist in the vocabulary."""
    special_tokens = ["<PAD>", "<UNK>", "<CLS>", "<SEP>", "<MASK>", "<START>", "<EOS>", "<EP-AVAILABLE>"]
    for t in special_tokens:
        assert t in tokenizer.vocab, f"Special token {t} not found in vocab"

def test_piece_tokens(tokenizer):
    """Test that both letter and symbol representations map to the same token ID."""
    # For example, white pawn 'P' and '♙' should map to the same ID
    piece_pairs = [
        ("P", "♙"), ("N", "♘"), ("B", "♗"), ("R", "♖"), ("Q", "♕"), ("K", "♔"),
        ("p", "♟"), ("n", "♞"), ("b", "♝"), ("r", "♜"), ("q", "♛"), ("k", "♚"),
        (".", "·")
    ]
    for letter, symbol in piece_pairs:
        assert tokenizer.vocab[letter] == tokenizer.vocab[symbol], f"Piece symbol {symbol} does not match ID of {letter}"

def test_decode_skip_special_tokens(tokenizer):
    """Test decoding with skip_special_tokens=True."""
    input_ids = [
        tokenizer.vocab["<CLS>"],
        tokenizer.vocab["♖"],
        tokenizer.vocab["<SEP>"],
        tokenizer.vocab["<EOS>"]
    ]
    decoded = tokenizer.decode(input_ids, skip_special_tokens=True)
    # Should only contain the piece symbols, skipping <CLS>, <SEP>, and <EOS>
    assert decoded.strip() == "♖"

def test_encode_fen_with_castling_and_enpassant(tokenizer):
    """Test encoding a position with specific castling rights and en passant square."""
    # Position after 1. e4 d5 2. exd5
    fen = "rnbqkbnr/ppp2ppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 3"
    encoded = tokenizer.encode_fen(fen=fen, return_tensors=None)
    
    tokens = [tokenizer.ids_to_tokens[i] for i in encoded["input_ids"]]
    
    # Check that <EP-AVAILABLE> token and the en passant square 'e3' are present
    # Also check castling rights tokens for both sides
    assert "<EP-AVAILABLE>" in tokens
    assert "e3" in tokens
    # White castling rights
    assert "<W-CASTLE-K>" in tokens
    assert "<W-CASTLE-Q>" in tokens
    # Black castling rights
    assert "<B-CASTLE-K>" in tokens
    assert "<B-CASTLE-Q>" in tokens

def test_encode_moves_with_promotion(tokenizer):
    """Test encoding moves that include a promotion."""
    # A promotion move, for instance "e7e8q"
    moves = "1. c4 e5 2. Nc3 Nc6 3. g3 g6 4. Bg2 Bg7 5. d3 d6 6. Rb1 f5 7. Bd2 Nf6 8. b4 O-O 9. b5 Ne7 10. Nf3 h6 11. O-O Be6 12. Qc1 Kh7 13. h3 Qc8 14. Kh2 Bf7 15. c5 g5 16. Qa3 g4 17. cxd6 gxf3 18. dxe7 fxg2 19. exf8=N+ Bxf8 20. Qa4 gxf1=N+"
    encoded = tokenizer(moves=moves, return_tensors=None, padding=False)
    
    tokens = [tokenizer.ids_to_tokens[i] for i in encoded["input_ids"]]
    
    # Check that promotion tokens are present for promoted knight
    # We expect <PROMOTE-N> whenever a piece is promoted to a knight
    assert "<PROMOTE-N>" in tokens

def test_empty_input(tokenizer):
    """Test encoding an empty input for moves."""
    encoded = tokenizer(moves="", return_tensors=None, padding=False)
    tokens = [tokenizer.ids_to_tokens[i] for i in encoded["input_ids"]]
    # For empty moves, it should return ["<START>"] and possibly no <EOS>
    assert "<START>" in tokens
    assert "<UNK>" not in tokens  # Should not produce unknown tokens for empty input

def test_game_over_scenario(tokenizer):
    """Test adding EOS token at game end. For example, a position that is checkmate or no legal moves."""

    moves = "1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7#"
    encoded = tokenizer(moves=moves, return_tensors=None, padding=False, add_eos=False)
    tokens = [tokenizer.ids_to_tokens[i] for i in encoded["input_ids"]]

    assert tokens[-1] == "<EOS>", "No EOS token at the end of the final move sequence"


    moves = "e2e4 e7e5 g1f3 b8c6"
    encoded = tokenizer(moves=moves, return_tensors=None, padding=False, add_eos=False)
    tokens = [tokenizer.ids_to_tokens[i] for i in encoded["input_ids"]]
    assert tokens[-1] != "<EOS>", "No EOS token at the end of the seq that is not game over"

    moves = "e2e4 e7e5 g1f3 b8c6"
    encoded = tokenizer(moves=moves, return_tensors=None, padding=False) # add_eos=True by default
    tokens = [tokenizer.ids_to_tokens[i] for i in encoded["input_ids"]]
    assert tokens[-1] == "<EOS>", "No EOS token at the end of the seq that is not game over"
    


def test_truncation(tokenizer):
    """Test truncation by specifying a max_length smaller than the full sequence."""
    fen = chess.STARTING_FEN
    moves = "e2e4 e7e5 g1f3 b8c6"
    # Normally this would produce a sequence longer than 10 tokens
    encoded = tokenizer(fens=fen, moves=moves, max_length=10, truncation=True)
    assert encoded["input_ids"].size(1) == 10, "Sequence was not truncated to max_length=10"

def test_padding(tokenizer):
    """Test padding by specifying a max_length larger than the sequence length."""
    fen = chess.STARTING_FEN
    moves = "e2e4 e7e5"
    # Without padding, let's see the sequence length
    no_pad_encoded = tokenizer(fens=fen, moves=moves, padding=False)
    seq_len = no_pad_encoded["input_ids"].size(1)
    
    # Now with padding to length seq_len+10
    padded_encoded = tokenizer(fens=fen, moves=moves, padding=True, max_length=seq_len+10)
    assert padded_encoded["input_ids"].size(1) == seq_len + 10, "Sequence was not padded correctly"

def test_save_and_load_vocabulary(tmp_path, tokenizer):
    """Test saving and loading the vocabulary."""
    save_dir = tmp_path / "tokenizer_test"
    save_dir.mkdir(exist_ok=True)
    tokenizer.save_vocabulary(str(save_dir))
    
    loaded_tokenizer = ChessTokenizer(vocab_file=str(save_dir / "chess_vocab.json"))
    assert loaded_tokenizer.vocab == tokenizer.vocab, "Loaded vocabulary does not match the original one"

def test_encode_game_at_position(tokenizer):
    """Test encoding game at a specific position index."""
    moves = "e2e4 e7e5 g1f3 b8c6"
    # Position index 2 means after 'e2e4 e7e5', we encode position at that point
    encoded = tokenizer.encode_game_at_position(moves, position_idx=2, add_special_tokens=True, return_tensors="pt", max_length=100)
    
    # The input should start with <CLS> and represent the position after e4, e5
    # Then followed by <SEP> and the moves g1f3 b8c6 <EOS>
    tokens = tokenizer.decode(encoded["input_ids"][0])
    assert "<CLS>" in tokens and "<SEP>" in tokens, "Expected CLS and SEP tokens in encoded output"
    assert "g1" in tokens and "f3" in tokens, "Expected next moves to be tokenized properly"
    assert "<EOS>" in tokens, "Expected EOS token after final move"

def test_encode_game_at_position_formats(tokenizer):
    """Test different move format inputs for encode_game_at_position."""
    # Test UCI move list
    moves_list = ["d2d4", "f7f5", "g2g3", "g7g6", "f1g2", "f8g7", "g1f3"]
    encoded_list = tokenizer.encode_game_at_position(
        moves=moves_list,
        position_idx=3,
        return_tensors="pt"
    )
    
    # Test UCI move string
    moves_str = "d2d4 f7f5 g2g3 g7g6 f1g2 f8g7 g1f3"
    encoded_str = tokenizer.encode_game_at_position(
        moves=moves_str,
        position_idx=3,
        return_tensors="pt"
    )
    
    # Test PGN format
    pgn_str = "1. d4 f5 2. g3 g6 3. Bg2 Bg7 4. Nf3"
    encoded_pgn = tokenizer.encode_game_at_position(
        moves=pgn_str,
        position_idx=3,
        return_tensors="pt"
    )
    
    # All formats should produce the same encoding for equivalent moves
    assert torch.equal(encoded_list["input_ids"], encoded_str["input_ids"])
    assert torch.equal(encoded_list["input_ids"], encoded_pgn["input_ids"])

def test_encode_game_at_position_custom_start(tokenizer):
    """Test encoding from a custom starting position."""
    custom_fen = "r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2R1/PqP2bPP/7K b - - 0 24"
    moves = "f2g3 e6e7 b2b1 b3c1 b1c1 h6c1"
    
    encoded = tokenizer.encode_game_at_position(
        moves=moves,
        position_idx=2,
        starting_fen=custom_fen,
        return_tensors="pt"
    )
    
    # Decode the tokens to verify position and remaining moves
    tokens = tokenizer.decode(encoded["input_ids"][0]).split()
    
    # Check for required position elements
    assert "♜" in tokens  # Bishop should be present
    assert "<START>" in tokens  # Start token for moves
    assert "h6" in tokens  # Part of remaining moves


def test_encode_game_at_position_boundaries(tokenizer):
    """Test boundary conditions for position index."""
    moves = ["e2e4", "e7e5", "g1f3", "b8c6"]
    
    # Test start position (idx = 0)
    start_pos = tokenizer.encode_game_at_position(
        moves=moves,
        position_idx=0,
        return_tensors="pt"
    )
    assert start_pos["input_ids"].size(1) > 0
    
    # Test end position (idx = len(moves))
    end_pos = tokenizer.encode_game_at_position(
        moves=moves,
        position_idx=len(moves),
        return_tensors="pt"
    )
    assert end_pos["input_ids"].size(1) > 0
    
    # Test invalid index
    with pytest.raises(ValueError, match="Invalid position_idx"):
        tokenizer.encode_game_at_position(
            moves=moves,
            position_idx=len(moves) + 1,
            return_tensors="pt"
        )

def test_encode_game_at_position_truncation_padding(tokenizer):
    """Test truncation and padding functionality."""
    moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5"]
    
    # Test truncation
    encoded_truncated = tokenizer.encode_game_at_position(
        moves=moves,
        position_idx=2,
        max_length=10,
        truncation=True,
        return_tensors="pt"
    )
    assert encoded_truncated["input_ids"].size(1) == 10
    
    # Test padding
    encoded_padded = tokenizer.encode_game_at_position(
        moves=moves,
        position_idx=2,
        max_length=100,
        padding=True,
        return_tensors="pt"
    )
    assert encoded_padded["input_ids"].size(1) == 100
    assert torch.any(encoded_padded["input_ids"] == tokenizer.pad_token_id)

def test_encode_game_at_position_special_tokens(tokenizer):
    """Test special token handling."""
    moves = ["e2e4", "e7e5", "g1f3", "b8c6"]
    
    # Test with special tokens
    encoded_with_special = tokenizer.encode_game_at_position(
        moves=moves,
        position_idx=2,
        add_special_tokens=True,
        return_tensors="pt"
    )
    tokens_with_special = tokenizer.decode(encoded_with_special["input_ids"][0]).split()
    assert "<CLS>" in tokens_with_special
    assert "<SEP>" in tokens_with_special
    
    # Test without special tokens
    encoded_without_special = tokenizer.encode_game_at_position(
        moves=moves,
        position_idx=2,
        add_special_tokens=False,
        return_tensors="pt"
    )
    tokens_without_special = tokenizer.decode(encoded_without_special["input_ids"][0]).split()
    assert "<CLS>" not in tokens_without_special
    assert "<SEP>" not in tokens_without_special

def test_encode_game_at_position_game_end(tokenizer):
    """Test handling of game-ending positions."""
    # Fool's mate
    moves = ["f2f3", "e7e5", "g2g4", "d8h4"]
    
    encoded = tokenizer.encode_game_at_position(
        moves=moves,
        position_idx=2,
        add_eos=True,
        return_tensors="pt"
    )
    
    tokens = tokenizer.decode(encoded["input_ids"][0]).split()
    assert "<EOS>" in tokens  # Should contain EOS token due to checkmate
    
    # Test without EOS token
    encoded_no_eos = tokenizer.encode_game_at_position(
        moves=moves,
        position_idx=2,
        add_eos=False,
        return_tensors="pt"
    )
    
    tokens_no_eos = tokenizer.decode(encoded_no_eos["input_ids"][0]).split()
    assert tokens_no_eos.count("<EOS>") == 1  # Should still have EOS due to game end

def test_encode_game_at_position_invalid_moves(tokenizer):
    """Test handling of invalid moves."""
    # Test invalid UCI move
    with pytest.raises(ValueError, match="Error parsing moves"):
        tokenizer.encode_game_at_position(
            moves=["e2e4", "invalid", "g1f3"],
            position_idx=1
        )
    
    # Test illegal move
    with pytest.raises(ValueError, match="Illegal move"):
        tokenizer.encode_game_at_position(
            moves=["e2e4", "e7e5", "e2e4"],  # e2e4 is illegal after e7e5
            position_idx=1
        )

