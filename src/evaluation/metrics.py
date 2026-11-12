"""
src/evaluation/metrics.py
Single Responsibility: evaluation metrics only.
No imports from training or demo modules — zero circular dependencies.
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import chess

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.core.board_encoding import board_to_tensor, legal_move_mask


@torch.no_grad()
def top_k_accuracy(model, dataset, k_values=(1, 3, 5), device="cpu", batch_size=256) -> dict:
    """Standard imitation-learning metric: does top-k include the human's move?"""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    correct = {k: 0 for k in k_values}
    total = 0
    for batch in loader:
        board = batch["board"].to(device)
        move_idx = batch["move_idx"].to(device)
        history = batch["history"].to(device)
        logits, _ = model(board, history_tensor=history)
        for k in k_values:
            topk = logits.topk(k, dim=1).indices
            correct[k] += (topk == move_idx.unsqueeze(1)).any(dim=1).sum().item()
        total += board.size(0)
    return {f"top_{k}_acc": correct[k] / total for k in k_values}


@torch.no_grad()
def style_consistency(model, boards_fens: list[str],
                      history_a: list, history_b: list, device="cpu") -> float:
    """
    Average KL divergence between two style-conditioned policies across positions.
    Near 0 → model ignores style. Larger → model genuinely conditions on style.
    KL is computed only over legal-move indices to avoid NaN from -inf logits.
    """
    kls = []
    for fen in boards_fens:
        board = chess.Board(fen)
        board_t = torch.from_numpy(board_to_tensor(board)).unsqueeze(0).to(device)
        mask, _ = legal_move_mask(board)
        mask_t = torch.from_numpy(mask).to(device)
        legal_idx = mask_t.nonzero(as_tuple=True)[0]

        hist_a = torch.tensor([history_a], dtype=torch.float32).to(device)
        hist_b = torch.tensor([history_b], dtype=torch.float32).to(device)

        logits_a, _ = model(board_t, history_tensor=hist_a)
        logits_b, _ = model(board_t, history_tensor=hist_b)

        la = logits_a.squeeze(0)[legal_idx]
        lb = logits_b.squeeze(0)[legal_idx]

        pa = F.log_softmax(la, dim=0)
        pb = F.softmax(lb, dim=0)
        kl = F.kl_div(pa, pb, reduction="sum", log_target=False)
        kls.append(kl.item())

    return sum(kls) / len(kls) if kls else 0.0
