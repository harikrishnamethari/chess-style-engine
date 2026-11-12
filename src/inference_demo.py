"""
src/inference_demo.py
CLI demo: shows top-k moves under style A / B / blend for any FEN.
Uses the history encoder — no more static player_id lookup table.
"""
import argparse
import sys, os
import torch
import chess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.board_encoding import board_to_tensor, legal_move_mask
from src.utils.checkpoint import load_checkpoint


def top_k_moves(model, board: chess.Board, history: list, k: int = 5, device: str = "cpu"):
    board_t = torch.from_numpy(board_to_tensor(board)).unsqueeze(0).to(device)
    mask, idx_to_move = legal_move_mask(board)
    mask_t = torch.from_numpy(mask).to(device)
    hist_t = torch.tensor([history], dtype=torch.float32).to(device)

    with torch.no_grad():
        logits, blunder_logit = model(board_t, history_tensor=hist_t)
        logits = logits.squeeze(0)
        masked = logits.masked_fill(mask_t == 0, float("-inf"))
        probs = torch.softmax(masked, dim=0)

    top_probs, top_idx = torch.topk(probs, k)
    moves = [(idx_to_move[i].uci() if i in idx_to_move else f"<{i}>", p)
             for i, p in zip(top_idx.tolist(), top_probs.tolist())]
    blunder_p = torch.sigmoid(blunder_logit).item() if blunder_logit is not None else None
    return moves, blunder_p


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_checkpoint(args.ckpt, device=device)
    board = chess.Board(args.fen) if args.fen else chess.Board()

    # Two contrasting style histories for demonstration
    aggressive_history = [[60.0, 0.25, 30.0]] * 10   # high CP loss, frequent captures, late castling
    passive_history    = [[20.0, 0.08, 12.0]] * 10   # low CP loss, avoids captures, early castling
    blend_history      = [[(a + b) / 2 for a, b in zip(ag, pa)]
                          for ag, pa in zip(aggressive_history, passive_history)]

    print(f"Position: {board.fen()}\n")
    for label, hist in [
        ("Style A — Aggressive", aggressive_history),
        ("Style B — Passive",    passive_history),
        ("50/50 Blend A+B",      blend_history),
    ]:
        moves, blunder_p = top_k_moves(model, board, hist, k=args.k, device=device)
        print(f"-- {label} --")
        for uci, p in moves:
            print(f"   {uci}  p={p:.3f}")
        if blunder_p is not None:
            print(f"   [blunder risk: {blunder_p:.3f}]")
        print()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="chess_style_net.pt")
    p.add_argument("--fen", default=None)
    p.add_argument("--k", type=int, default=5)
    main(p.parse_args())
