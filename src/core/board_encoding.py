"""
Board <-> tensor encoding, and Move <-> class-index encoding.

Board encoding: 18 x 8 x 8 tensor
  planes 0-5   : white pieces (P N B R Q K)
  planes 6-11  : black pieces (p n b r q k)
  plane 12     : side to move (all 1s if white to move, else 0s)
  planes 13-16 : castling rights (white kingside, white queenside,
                                   black kingside, black queenside)
  plane 17     : en passant target square (1 at that square, else 0)

Move encoding: AlphaZero-style 8x8x73 move plane -> flattened to 4672 classes.
  index = from_square * 73 + plane
  planes 0-55  : "queen-like" moves: 8 directions x 7 distances
  planes 56-63 : knight moves (8 L-shaped deltas)
  planes 64-72 : underpromotions: 3 directions (fwd, capture-left, capture-right)
                 x 3 piece types (knight, bishop, rook). Queen promotions are
                 encoded as normal queen-like moves (distance 1, diagonal/forward).

This mirrors the encoding used in the AlphaZero paper (Silver et al., 2017),
adapted here for supervised imitation learning rather than self-play RL.
"""

import chess
import numpy as np

NUM_MOVE_PLANES = 73
NUM_MOVE_CLASSES = 64 * NUM_MOVE_PLANES  # 4672

# 8 queen-like directions as (delta_file, delta_rank), clockwise from North
_QUEEN_DIRS = [
    (0, 1),   # N
    (1, 1),   # NE
    (1, 0),   # E
    (1, -1),  # SE
    (0, -1),  # S
    (-1, -1), # SW
    (-1, 0),  # W
    (-1, 1),  # NW
]

# 8 knight deltas
_KNIGHT_DIRS = [
    (1, 2), (2, 1), (2, -1), (1, -2),
    (-1, -2), (-2, -1), (-2, 1), (-1, 2),
]

# underpromotion pieces (queen handled as a normal queen-like move)
_UNDERPROMO_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]
# underpromotion directions relative to pawn's forward direction: (fwd, capture-left, capture-right)
_UNDERPROMO_DELTAS = [(0, 1), (-1, 1), (1, 1)]  # rank delta always "forward"; sign flipped for black below


def board_to_tensor(board: chess.Board) -> np.ndarray:
    """Encode a python-chess Board into an (18, 8, 8) float32 tensor."""
    t = np.zeros((18, 8, 8), dtype=np.float32)
    piece_plane = {
        (chess.PAWN, chess.WHITE): 0, (chess.KNIGHT, chess.WHITE): 1,
        (chess.BISHOP, chess.WHITE): 2, (chess.ROOK, chess.WHITE): 3,
        (chess.QUEEN, chess.WHITE): 4, (chess.KING, chess.WHITE): 5,
        (chess.PAWN, chess.BLACK): 6, (chess.KNIGHT, chess.BLACK): 7,
        (chess.BISHOP, chess.BLACK): 8, (chess.ROOK, chess.BLACK): 9,
        (chess.QUEEN, chess.BLACK): 10, (chess.KING, chess.BLACK): 11,
    }
    for square, piece in board.piece_map().items():
        rank, file = chess.square_rank(square), chess.square_file(square)
        plane = piece_plane[(piece.piece_type, piece.color)]
        t[plane, rank, file] = 1.0

    if board.turn == chess.WHITE:
        t[12, :, :] = 1.0

    t[13, :, :] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    t[14, :, :] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    t[15, :, :] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    t[16, :, :] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0

    if board.ep_square is not None:
        r, f = chess.square_rank(board.ep_square), chess.square_file(board.ep_square)
        t[17, r, f] = 1.0

    return t


def move_to_index(move: chess.Move, board: chess.Board) -> int:
    """Map a legal chess.Move (in the context of `board`) to a class index in [0, 4671]."""
    from_sq = move.from_square
    to_sq = move.to_square
    from_file, from_rank = chess.square_file(from_sq), chess.square_rank(from_sq)
    to_file, to_rank = chess.square_file(to_sq), chess.square_rank(to_sq)
    d_file, d_rank = to_file - from_file, to_rank - from_rank

    piece = board.piece_at(from_sq)
    is_pawn = piece is not None and piece.piece_type == chess.PAWN

    # underpromotion (knight/bishop/rook only; queen promo falls through to queen-like branch)
    if move.promotion is not None and move.promotion != chess.QUEEN:
        color_sign = 1 if piece.color == chess.WHITE else -1
        rel_d_file = d_file  # -1, 0, +1
        # direction index: 0=forward, 1=capture-left, 2=capture-right
        if rel_d_file == 0:
            dir_idx = 0
        elif rel_d_file == -color_sign:
            dir_idx = 1
        else:
            dir_idx = 2
        piece_idx = _UNDERPROMO_PIECES.index(move.promotion)
        plane = 64 + dir_idx * 3 + piece_idx
        return from_sq * NUM_MOVE_PLANES + plane

    # knight move
    if (d_file, d_rank) in _KNIGHT_DIRS or (piece is not None and piece.piece_type == chess.KNIGHT):
        if (d_file, d_rank) in _KNIGHT_DIRS:
            k_idx = _KNIGHT_DIRS.index((d_file, d_rank))
            plane = 56 + k_idx
            return from_sq * NUM_MOVE_PLANES + plane

    # queen-like move (includes queen promotions, king moves, rook/bishop/queen moves, single pawn steps)
    step_file = 0 if d_file == 0 else (1 if d_file > 0 else -1)
    step_rank = 0 if d_rank == 0 else (1 if d_rank > 0 else -1)
    distance = max(abs(d_file), abs(d_rank))
    dir_idx = _QUEEN_DIRS.index((step_file, step_rank))
    plane = dir_idx * 7 + (distance - 1)
    return from_sq * NUM_MOVE_PLANES + plane


def legal_move_mask(board: chess.Board) -> np.ndarray:
    """Return a (4672,) 0/1 mask over legal moves for the current position."""
    mask = np.zeros(NUM_MOVE_CLASSES, dtype=np.float32)
    idx_to_move = {}
    for mv in board.legal_moves:
        idx = move_to_index(mv, board)
        mask[idx] = 1.0
        idx_to_move[idx] = mv
    return mask, idx_to_move


if __name__ == "__main__":
    # sanity check: every legal move in the starting position round-trips uniquely
    board = chess.Board()
    seen = {}
    for mv in board.legal_moves:
        idx = move_to_index(mv, board)
        assert idx not in seen, f"collision: {mv} and {seen[idx]} both map to {idx}"
        seen[idx] = mv
    print(f"OK: {len(seen)} legal opening moves mapped to unique indices out of {NUM_MOVE_CLASSES} classes")
    print("board tensor shape:", board_to_tensor(board).shape)
