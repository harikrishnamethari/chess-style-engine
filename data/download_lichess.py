"""
Real data pipeline for training on actual human games. Run this in Colab
(or any environment with general internet access) -- it downloads a Lichess
monthly PGN dump and extracts (fen, move_uci, player_id) rows for a chosen
set of players, split by rating band so "player_id" doubles as a rating-band
style label if you don't want to chase specific named players.

Usage (in Colab):
    !pip install python-chess zstandard requests -q
    python download_lichess.py --year 2024 --month 1 --max_games 20000 \
        --out lichess_sample.csv

Notes:
  - Lichess monthly dumps are large (many GB compressed). Use --max_games to
    cap how many games you actually parse -- 20k-50k games is plenty for a
    day-long training run on a free Colab GPU.
  - This script assigns player_id by RATING BAND by default (e.g. "1000-1200",
    "2000-2200") since that's guaranteed to have enough games per bucket and
    gives you a clean "style" story: skill-band-conditioned play.
  - To condition on SPECIFIC named players instead (e.g. "play like Hikaru"),
    pass --players "DrNykterstein,Hikaru" -- but note famous players'
    games are a much smaller slice of the dump, so you may need more months
    or Lichess's per-player game export endpoint instead
    (https://lichess.org/api/games/user/<username>).
"""

import argparse
import io
import os
import sys
import csv
import requests
import chess
import chess.pgn

try:
    import zstandard as zstd
except ImportError:
    zstd = None


RATING_BANDS = [(0, 1200), (1200, 1600), (1600, 2000), (2000, 4000)]


def rating_band_label(rating):
    if rating is None:
        return None
    for lo, hi in RATING_BANDS:
        if lo <= rating < hi:
            return f"{lo}-{hi}"
    return None


def stream_lichess_pgn(year, month):
    """Streams the decompressed PGN text for a Lichess monthly standard-rated dump."""
    if zstd is None:
        raise RuntimeError("pip install zstandard first")
    url = f"https://database.lichess.org/standard/lichess_db_standard_rated_{year}-{month:02d}.pgn.zst"
    print(f"Streaming {url} ...")
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    dctx = zstd.ZstdDecompressor()
    stream_reader = dctx.stream_reader(resp.raw)
    text_stream = io.TextIOWrapper(stream_reader, encoding="utf-8")
    return text_stream


def extract_rows(pgn_stream, max_games, players=None):
    rows = []
    games_parsed = 0
    player_set = set(p.lower() for p in players) if players else None

    while games_parsed < max_games:
        game = chess.pgn.read_game(pgn_stream)
        if game is None:
            break
        games_parsed += 1

        white = game.headers.get("White", "")
        black = game.headers.get("Black", "")
        white_elo = game.headers.get("WhiteElo")
        black_elo = game.headers.get("BlackElo")
        try:
            white_elo = int(white_elo) if white_elo and white_elo != "?" else None
            black_elo = int(black_elo) if black_elo and black_elo != "?" else None
        except ValueError:
            white_elo = black_elo = None

        board = game.board()
        for move in game.mainline_moves():
            mover_is_white = board.turn == chess.WHITE

            if player_set is not None:
                name = white if mover_is_white else black
                if name.lower() not in player_set:
                    board.push(move)
                    continue
                pid = name
            else:
                elo = white_elo if mover_is_white else black_elo
                pid = rating_band_label(elo)
                if pid is None:
                    board.push(move)
                    continue

            rows.append({"fen": board.fen(), "move_uci": move.uci(), "player_id": pid})
            board.push(move)

        if games_parsed % 1000 == 0:
            print(f"  parsed {games_parsed} games, {len(rows)} rows so far")

    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--month", type=int, required=True)
    p.add_argument("--max_games", type=int, default=20000)
    p.add_argument("--players", type=str, default=None, help="comma-separated Lichess usernames")
    p.add_argument("--out", type=str, default="lichess_sample.csv")
    args = p.parse_args()

    players = args.players.split(",") if args.players else None
    pgn_stream = stream_lichess_pgn(args.year, args.month)
    rows = extract_rows(pgn_stream, args.max_games, players=players)

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["fen", "move_uci", "player_id"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
