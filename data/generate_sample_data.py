"""
Generates a small synthetic dataset of (fen_before_move, move_uci, fake_player_id)
triples so the training pipeline can be validated end-to-end on CPU, with zero
external dependencies. This is NOT real style data -- it exists purely to catch
bugs before you spend Colab GPU time on the real Lichess pipeline.

Two synthetic "styles" are simulated to prove the conditioning mechanism works:
  - player 0 ("aggressive"): prefers moves that step into the center / capture
  - player 1 ("passive"): prefers the shortest-distance / most conservative move
This lets us later verify the model actually learns *different* distributions
per style, not just a single averaged policy.
"""

import csv
import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import chess


def score_aggressive(board, move):
    score = 0
    if board.is_capture(move):
        score += 10
    to_file, to_rank = chess.square_file(move.to_square), chess.square_rank(move.to_square)
    center_dist = abs(3.5 - to_file) + abs(3.5 - to_rank)
    score += (7 - center_dist)
    return score


def score_passive(board, move):
    from_file, from_rank = chess.square_file(move.from_square), chess.square_rank(move.from_square)
    to_file, to_rank = chess.square_file(move.to_square), chess.square_rank(move.to_square)
    dist = abs(to_file - from_file) + abs(to_rank - from_rank)
    score = -dist
    if board.is_capture(move):
        score -= 5  # passive player avoids trades
    return score


def simulate_game(style_fn, max_plies=40, seed=None):
    rng = random.Random(seed)
    board = chess.Board()
    records = []
    for _ in range(max_plies):
        if board.is_game_over():
            break
        legal = list(board.legal_moves)
        scored = [(style_fn(board, mv), mv) for mv in legal]
        scored.sort(key=lambda x: x[0], reverse=True)
        # pick from top-3 with some randomness so games aren't identical
        top = scored[: min(3, len(scored))]
        mv = rng.choice(top)[1]
        
        # Simulate eval drop for blunder prediction
        # Most moves have small drop, ~5% are blunders (>150cp drop)
        is_blunder = rng.random() < 0.05
        eval_drop = rng.randint(151, 400) if is_blunder else rng.randint(0, 50)
        
        records.append((board.fen(), mv.uci(), eval_drop))
        board.push(mv)
    return records


def main(out_path, n_games_per_style=40):
    import json
    styles = {0: score_aggressive, 1: score_passive}
    rows = []
    for player_id, style_fn in styles.items():
        for g in range(n_games_per_style):
            for fen, uci, eval_drop in simulate_game(style_fn, seed=g * 7 + player_id):
                rows.append({"fen": fen, "move_uci": uci, "player_id": player_id, "eval_drop": eval_drop})

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["fen", "move_uci", "player_id", "eval_drop"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} (fen, move, player_id, eval_drop) rows to {out_path}")
    print("player_id 0 = 'aggressive' style, player_id 1 = 'passive' style")
    
    # Generate mock historical stats for each player ID (shape: N_games x num_features)
    # Features: [avg_cp_loss, capture_rate, castling_timing]
    # We will simulate N=10 games history.
    stats = {}
    rng = random.Random(42)
    for pid in styles.keys():
        player_history = []
        for _ in range(10):
            if pid == 0: # aggressive
                hist = [rng.uniform(30, 80), rng.uniform(0.15, 0.3), rng.uniform(20, 40)]
            else: # passive
                hist = [rng.uniform(10, 40), rng.uniform(0.05, 0.15), rng.uniform(10, 20)]
            player_history.append(hist)
        stats[str(pid)] = player_history
    
    stats_path = out_path.replace(".csv", "_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Wrote mock historical stats to {stats_path}")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "sample_data.csv"
    main(out)
