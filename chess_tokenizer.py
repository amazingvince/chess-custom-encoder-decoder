from transformers import PreTrainedTokenizer
import chess
import chess.svg
import cairosvg


class ChessTokenizer(PreTrainedTokenizer):
    def __init__(self, 
                 square_tokens=None, 
                 promotion_tokens=None,
                 fen_special_tokens=None,
                 special_tokens=None,
                 piece_symbols=None,
                 default_start_fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                 **kwargs):

        # Default tokens
        if square_tokens is None:
            files = 'abcdefgh'
            ranks = '12345678'
            square_tokens = [f"{f}{r}" for f in files for r in ranks]

        if promotion_tokens is None:
            promotion_tokens = ["=Q", "=R", "=B", "=N"]

        # Add tokens to represent which side is to move
        if fen_special_tokens is None:
            fen_special_tokens = ["<FEN>", "<FEN_END>", "<RANKSEP>", "<SPACE>", "<NOCASTLE>", "<WHITE_TO_MOVE>", "<BLACK_TO_MOVE>"]

        if special_tokens is None:
            special_tokens = ["<START>", "<END>", "<PAD>", "<MASK>"]

        if piece_symbols is None:
            piece_symbols = list("PNBRQKpnbrqk_")

        self.default_start_fen = default_start_fen

        # Build vocab
        self.vocab = {}
        idx = 0

        # Squares
        for tok in square_tokens:
            self.vocab[tok] = idx
            idx += 1

        # Promotions
        for tok in promotion_tokens:
            self.vocab[tok] = idx
            idx += 1

        # FEN tokens (including <WHITE_TO_MOVE> and <BLACK_TO_MOVE>)
        for tok in fen_special_tokens:
            self.vocab[tok] = idx
            idx += 1

        # Piece symbols for fen encoding
        for sym in piece_symbols:
            if sym not in self.vocab:
                self.vocab[sym] = idx
                idx += 1

        # Special tokens
        for tok in special_tokens:
            self.vocab[tok] = idx
            idx += 1

        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}

        self.pad_token_id = self.vocab["<PAD>"]
        self.eos_token_id = self.vocab["<END>"]
        self.bos_token_id = self.vocab["<START>"]
        self.mask_token_id = self.vocab["<MASK>"]

        super().__init__(**kwargs)

    def get_vocab(self):
        return self.vocab


    def encode_fen(self, fen=None):
        # If no fen given, use default start position
        if fen is None:
            fen = self.default_start_fen

        board = chess.Board(fen)
        
        tokens = ["<FEN>"]

        # Piece placement
        for rank in range(7, -1, -1):
            rank_tokens = []
            for file in range(0, 8):
                square = rank * 8 + file
                piece = board.piece_at(square)
                if piece is None:
                    rank_tokens.append("_")
                else:
                    rank_tokens.append(piece.symbol())
            tokens.extend(rank_tokens)
            if rank > 0:
                tokens.append("<RANKSEP>")

        # Active color as a special token
        tokens.append("<SPACE>")
        if board.turn:
            tokens.append("<WHITE_TO_MOVE>")
        else:
            tokens.append("<BLACK_TO_MOVE>")

        # Castling rights
        tokens.append("<SPACE>")
        castling_str = ""
        if board.has_kingside_castling_rights(chess.WHITE):
            castling_str += "K"
        if board.has_queenside_castling_rights(chess.WHITE):
            castling_str += "Q"
        if board.has_kingside_castling_rights(chess.BLACK):
            castling_str += "k"
        if board.has_queenside_castling_rights(chess.BLACK):
            castling_str += "q"
        if castling_str == "":
            tokens.append("<NOCASTLE>")
        else:
            tokens.extend(list(castling_str))

        # En passant
        tokens.append("<SPACE>")
        if board.ep_square is None:
            tokens.append("-")
        else:
            ep_token = chess.square_name(board.ep_square)
            tokens.append(ep_token)

        # halfmove
        tokens.append("<SPACE>")
        tokens.append(str(board.halfmove_clock))
        # fullmove
        tokens.append("<SPACE>")
        tokens.append(str(board.fullmove_number))

        tokens.append("<FEN_END>")

        return self.convert_tokens_to_ids(tokens)

    def encode_moves(self, moves):
        tokens = []
        for move in moves:
            tokens.extend(move)
        return self.convert_tokens_to_ids(tokens)

    def convert_tokens_to_ids(self, tokens):
        return [self.vocab.get(t, self.mask_token_id) for t in tokens]

    def convert_ids_to_tokens(self, ids):
        return [self.ids_to_tokens[i] for i in ids]

    def build_inputs_with_special_tokens(self, fen_ids=None, move_ids=None):
        if fen_ids is None:
            fen_ids = self.encode_fen(None)  # Use default start position

        if move_ids is None:
            return [self.bos_token_id] + fen_ids + [self.eos_token_id]

        return [self.bos_token_id] + fen_ids + [self.eos_token_id, self.bos_token_id] + move_ids + [self.eos_token_id]

    def pad(self, encoded_inputs, max_length=None):
        if max_length is None:
            max_length = max(len(seq) for seq in encoded_inputs)
        padded = []
        for seq in encoded_inputs:
            if len(seq) < max_length:
                seq = seq + [self.pad_token_id]*(max_length - len(seq))
            padded.append(seq)
        return padded

    def vocab_size(self):
        return len(self.vocab)
    
    def generate_board_png(self, fen=None, filename="board.png"):
        """
        Generate a PNG image of the board from a given FEN.
        If fen is None, use the default starting fen.
        """
        if fen is None:
            fen = self.default_start_fen
        
        board = chess.Board(fen)
        # Generate SVG
        board_svg = chess.svg.board(board=board)
        
        # Convert SVG to PNG using cairosvg
        cairosvg.svg2png(bytestring=board_svg.encode('utf-8'), write_to=filename)
        print(f"Board image saved to {filename}")


if __name__ == "__main__":
    # Example usage:
    tokenizer = ChessTokenizer()

    fen_ids = tokenizer.encode_fen()  # default start fen
    print("FEN IDs (default):", fen_ids)
    print("FEN Tokens (default):", tokenizer.convert_ids_to_tokens(fen_ids))

    moves = [
        ("e2", "e4"),
        ("e7", "e5"),
        ("g1", "f3"),
        ("b8", "c6")
    ]
    move_ids = tokenizer.encode_moves(moves)
    print("Move IDs:", move_ids)
    print("Move Tokens:", tokenizer.convert_ids_to_tokens(move_ids))

    input_ids = tokenizer.build_inputs_with_special_tokens(fen_ids, move_ids)
    print("Combined Input IDs:", input_ids)
    print("Combined Tokens:", tokenizer.convert_ids_to_tokens(input_ids))

    # Print vocab size
    print("Vocab Size:", tokenizer.vocab_size())

    # Generate a board PNG
    tokenizer.generate_board_png()  # Will save board.png for the default start position

