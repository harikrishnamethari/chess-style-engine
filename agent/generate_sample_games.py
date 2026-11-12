"""
Generates full PGN games for a synthetic player who is deliberately weaker in
the endgame than the opening/middlegame (blunders more often as the game
progresses). This lets us verify the CoachAgent actually detects a
phase-specific weakness pattern rather than just finding random blunders.
"""

import random
import chess
import chess.pgn

# probability of playing a purely random (likely bad) move, by move-number phase
BLUNDER_PROB = {"opening": 0.05, "middlegame": 0.15, "endgame": 0.45}


def phase_for_ply(ply):
    if ply <= 10:
        return "opening"
    elif ply <= 30:
        return "middlegame"
    return "endgame"


def safe_ish_move(board):
    """Cheap heuristic: prefer captures, else prefer moves to squares not
    attacked by the opponent. Not a real engine -- just enough to look like
    non-random play so the injected endgame weakness stands out clearly."""
    legal = list(board.legal_moves)
    captures = [m for m in legal if board.is_capture(m)]
    if captures:
        return random.choice(captures)

    safe = []
    for m in legal:
        board.push(m)
        attacked = board.is_attacked_by(not board.turn, m.to_square)
        board.pop()
        if not attacked:
            safe.append(m)
    return random.choice(safe) if safe else random.choice(legal)


def simulate_weak_endgame_player_game(seed, max_plies=60, player_color=chess.WHITE):
    rng = random.Random(seed)
    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["White"] = "test_player" if player_color == chess.WHITE else "opponent"
    game.headers["Black"] = "opponent" if player_color == chess.WHITE else "test_player"
    node = game

    for ply in range(1, max_plies + 1):
        if board.is_game_over():
            break
        is_players_move = board.turn == player_color
        phase = phase_for_ply(ply)

        if is_players_move and rng.random() < BLUNDER_PROB[phase]:
            move = rng.choice(list(board.legal_moves))  # deliberate blunder
        else:
            move = safe_ish_move(board)

        node = node.add_variation(move)
        board.push(move)

    return game


def main(out_path, n_games=4):
    games = [simulate_weak_endgame_player_game(seed=i) for i in range(n_games)]
    with open(out_path, "w") as f:
        for g in games:
            f.write(str(g) + "\n\n")
    print(f"Wrote {n_games} synthetic games (player='test_player', weak in endgame) to {out_path}")


if __name__ == "__main__":
    import sys

    out = sys.argv[1] if len(sys.argv) > 1 else "sample_games.pgn"
    main(out)
