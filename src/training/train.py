"""
src/training/train.py  —  CLI entrypoint only.
Open/Closed principle: adding new training options never requires touching
Trainer; extend here by passing new args.
"""
import argparse
import sys
import os

# Allow running as `python src/training/train.py` from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.ml.model import ChessStyleNet
from src.ml.dataset import ChessStyleDataset
from src.training.trainer import Trainer
from src.utils.checkpoint import save_checkpoint


def main(args):
    device = args.device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
    dataset = ChessStyleDataset(args.csv)
    model = ChessStyleNet()
    trainer = Trainer(
        model=model,
        dataset=dataset,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
    )
    trainer.train(epochs=args.epochs)
    save_checkpoint(model, dataset.player_id_map, args.save_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train ChessStyleNet")
    p.add_argument("--csv", default="data/sample_data.csv")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--save_path", default="chess_style_net.pt")
    p.add_argument("--device", default=None)
    main(p.parse_args())
