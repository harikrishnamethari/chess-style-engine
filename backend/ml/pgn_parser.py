"""
backend/ml/pgn_parser.py
Single Responsibility: extract game statistics from PGN text.
No model or DB knowledge — pure data parsing.
"""
import io
import chess.pgn


def extract_game_metadata(pgn_text: str) -> dict:
    """Parse PGN and extract header metadata.
    Returns None if the text contains no valid game moves.
    """
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    # chess.pgn.read_game returns an empty game for garbage input;
    # require at least one move to consider it a real game.
    if game is None or game.next() is None:
        return None
    return {
        "white": game.headers.get("White", "?"),
        "black": game.headers.get("Black", "?"),
        "result": game.headers.get("Result", "?"),
        "event": game.headers.get("Event", "?"),
        "date": game.headers.get("Date", "?"),
    }


def extract_history_features(pgn_text: str, max_games: int = 10) -> list[list[float]]:
    """
    Extract a per-game feature vector from PGN(s).
    Features: [avg_cp_loss_proxy, capture_rate, castling_move_number]
    
    In production: parse Lichess %eval annotations for true CP loss.
    Here we approximate from move patterns.
    """
    history = []
    pgn_io = io.StringIO(pgn_text)

    while len(history) < max_games:
        game = chess.pgn.read_game(pgn_io)
        if game is None:
            break

        board = game.board()
        total_moves = 0
        captures = 0
        castling_move = 0
        move_num = 0

        for move in game.mainline_moves():
            move_num += 1
            if board.is_capture(move):
                captures += 1
            piece = board.piece_at(move.from_square)
            if piece and piece.piece_type == chess.KING:
                # Detect castling by king moving 2 squares
                df = abs(chess.square_file(move.to_square) - chess.square_file(move.from_square))
                if df == 2 and castling_move == 0:
                    castling_move = move_num
            board.push(move)
            total_moves += 1

        if total_moves == 0:
            continue

        capture_rate = captures / total_moves
        # Proxy for CP loss: late castling / no castling → likely more chaotic play
        castling_timing = castling_move if castling_move > 0 else 40
        # We don't have real eval, so use a simple heuristic
        avg_cp_loss_proxy = (1 - capture_rate) * 30 + (40 - castling_timing) * 0.5
        avg_cp_loss_proxy = max(5.0, min(100.0, avg_cp_loss_proxy))

        history.append([avg_cp_loss_proxy, capture_rate, float(castling_timing)])

    # Pad with neutral features if fewer than max_games
    neutral = [25.0, 0.12, 15.0]
    while len(history) < max_games:
        history.append(neutral)

    return history
