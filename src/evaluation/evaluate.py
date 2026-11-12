"""
src/evaluation/evaluate.py  —  CLI entrypoint for evaluation.
"""
import argparse
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.ml.dataset import ChessStyleDataset
from src.utils.checkpoint import load_checkpoint
from src.evaluation.metrics import top_k_accuracy, style_consistency


def main(args):
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    model, player_id_map = load_checkpoint(args.ckpt, device=device)
    ds = ChessStyleDataset(args.csv, player_id_map=player_id_map)

    print("=== Top-k Accuracy ===")
    metrics = top_k_accuracy(model, ds, device=device)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\n=== Style Consistency (KL Divergence) ===")
    pids = list(ds.player_stats.keys())
    if len(pids) >= 2:
        hist_a = ds.player_stats[pids[0]]
        hist_b = ds.player_stats[pids[1]]
        sample_fens = [ds.rows[i]["fen"] for i in range(min(50, len(ds)))]
        kl = style_consistency(model, sample_fens, hist_a, hist_b, device=device)
        print(f"  Avg KL (style '{pids[0]}' vs '{pids[1]}'): {kl:.4f}")
        print("  (Higher = model uses style conditioning; near 0 = ignoring it)")
    else:
        print("  Need at least 2 player styles in the dataset.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="chess_style_net.pt")
    p.add_argument("--csv", default="data/sample_data.csv")
    main(p.parse_args())
