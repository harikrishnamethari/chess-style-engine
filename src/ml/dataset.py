import csv
import json
import os
import chess
import torch
from torch.utils.data import Dataset

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.core.board_encoding import board_to_tensor, move_to_index


class ChessStyleDataset(Dataset):
    """
    Loads a CSV with columns: fen, move_uci, player_id, eval_drop
    Each row is one (position, move-the-human-played, who-played-it) example.
    Also loads _stats.json for historical features per player.
    """

    def __init__(self, csv_path: str, player_id_map: dict = None):
        self.rows = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                self.rows.append(r)

        if player_id_map is None:
            unique_ids = sorted(set(r["player_id"] for r in self.rows), key=lambda x: (len(x), x))
            player_id_map = {pid: i for i, pid in enumerate(unique_ids)}
        self.player_id_map = player_id_map
        self.num_players = len(player_id_map)

        stats_path = csv_path.replace(".csv", "_stats.json")
        self.player_stats = {}
        if os.path.exists(stats_path):
            with open(stats_path, "r") as f:
                self.player_stats = json.load(f)
        else:
            print(f"Warning: no stats file found at {stats_path}. Model training using HistoryEncoder will fail if stats are missing.")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        board = chess.Board(row["fen"])
        move = chess.Move.from_uci(row["move_uci"])

        board_tensor = torch.from_numpy(board_to_tensor(board))
        move_idx = move_to_index(move, board)
        player_id_raw = row["player_id"]
        player_id = self.player_id_map[player_id_raw]

        # blunder label: 1.0 if eval_drop > 150, else 0.0
        eval_drop = float(row.get("eval_drop", 0))
        blunder = 1.0 if eval_drop > 150 else 0.0

        # history tensor: shape (seq_len, num_features)
        # default to zero tensor if not found
        if player_id_raw in self.player_stats:
            history = torch.tensor(self.player_stats[player_id_raw], dtype=torch.float32)
        else:
            history = torch.zeros((10, 3), dtype=torch.float32)

        return {
            "board": board_tensor,
            "move_idx": torch.tensor(move_idx, dtype=torch.long),
            "player_id": torch.tensor(player_id, dtype=torch.long),
            "blunder": torch.tensor([blunder], dtype=torch.float32),
            "history": history,
        }


if __name__ == "__main__":
    ds = ChessStyleDataset("../data/sample_data.csv")
    print(f"num_examples={len(ds)} num_players={ds.num_players} player_id_map={ds.player_id_map}")
    sample = ds[0]
    print({k: v.shape for k, v in sample.items()})
