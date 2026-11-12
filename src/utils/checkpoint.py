"""
src/utils/checkpoint.py
Single Responsibility: all model checkpoint save/load logic lives here.
DRY: used by trainer, evaluate, and inference_demo — no duplication.
"""
import torch
from pathlib import Path


def save_checkpoint(model, player_id_map: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "player_id_map": player_id_map,
        },
        path,
    )
    print(f"Checkpoint saved → {path}")


def load_checkpoint(path: str, device: str = "cpu"):
    """
    Returns (model, player_id_map).
    Raises FileNotFoundError with a helpful message if the checkpoint is missing.
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from ml.model import ChessStyleNet

    if not Path(path).exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            "Run `python src/training/train.py` first to produce a checkpoint."
        )
    ckpt = torch.load(path, map_location=device)
    model = ChessStyleNet().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt["player_id_map"]
