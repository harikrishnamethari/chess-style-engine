"""
Convert a raw PGN file (even without headers or exported from Lichess/Chess.com)
into training CSV: (fen, move_uci, player_id).
"""
import io
import csv
import argparse
import chess
import chess.pgn


def parse_pgn_to_csv(pgn_path: str, out_csv: str, player_name: str = "player", max_games: int = 5000):
    print(f"Reading {pgn_path} ...")
    with open(pgn_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    stream = io.StringIO(text)
    rows = []
    games_parsed = 0
    errors = 0

    while games_parsed < max_games:
        try:
            game = chess.pgn.read_game(stream)
        except Exception:
            errors += 1
            continue

        if game is None:
            break

        white = game.headers.get("White", "").strip()
        black = game.headers.get("Black", "").strip()

        # If headers exist and match player_name, train on that color only.
        # If headers are missing (e.g. raw SAN moves '? vs ?'), train on all moves.
        if white != "?" and white.lower() == player_name.lower():
            target_color = chess.WHITE
        elif black != "?" and black.lower() == player_name.lower():
            target_color = chess.BLACK
        else:
            target_color = None  # Include all moves

        board = game.board()
        for move in game.mainline_moves():
            if not board.is_game_over():
                if target_color is None or board.turn == target_color:
                    rows.append({
                        "fen": board.fen(),
                        "move_uci": move.uci(),
                        "player_id": player_name
                    })
            board.push(move)

        games_parsed += 1
        if games_parsed % 100 == 0:
            print(f"Parsed {games_parsed} games ({len(rows)} positions)...")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["fen", "move_uci", "player_id"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Successfully processed {games_parsed} games.")
    print(f"Total training positions extracted: {len(rows)} -> saved to {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pgn", default="data/my_games.pgn", help="Path to input PGN")
    parser.add_argument("--out", default="data/my_games.csv", help="Path to output CSV")
    parser.add_argument("--player", default="harikrishna", help="Player label")
    parser.add_argument("--max_games", type=int, default=5000)
    args = parser.parse_args()

    parse_pgn_to_csv(args.pgn, args.out, player_name=args.player, max_games=args.max_games)

