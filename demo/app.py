"""
Streamlit demo for the style-conditioned chess engine.

Run with:  streamlit run app.py
(from inside demo/, with a trained checkpoint at ../src/chess_style_net.pt
or pass --ckpt via the sidebar path input)

Shows:
  - a board (via chess.svg) for a chosen or default position
  - a slider to interpolate between two learned styles
  - the model's live top-5 move distribution as the slider moves
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
import chess
import chess.svg
import torch

from board_encoding import board_to_tensor, legal_move_mask
from model import ChessStyleNet


@st.cache_resource
def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = ChessStyleNet(num_players=ckpt["num_players"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt["player_id_map"]


def predict_top_moves(model, board, style_vec, k=5):
    board_t = torch.from_numpy(board_to_tensor(board)).unsqueeze(0)
    mask, idx_to_move = legal_move_mask(board)
    mask_t = torch.from_numpy(mask)
    with torch.no_grad():
        logits, blunder_logit = model(board_t, style_vec=style_vec)
        logits = logits.squeeze(0).masked_fill(mask_t == 0, float("-inf"))
        probs = torch.softmax(logits, dim=0)
    top_probs, top_idx = torch.topk(probs, min(k, mask.sum().astype(int)))
    moves = [(idx_to_move[i.item()], p.item()) for p, i in zip(top_probs, top_idx)]
    blunder_p = torch.sigmoid(blunder_logit).item() if blunder_logit is not None else None
    return moves, blunder_p


st.set_page_config(page_title="Style-Conditioned Chess Engine", layout="wide")
st.title("Style-Conditioned Chess Engine")
st.caption(
    "One model, many playing styles. Move the slider to blend between two learned "
    "styles and watch the predicted move distribution shift in real time."
)

ckpt_path = st.sidebar.text_input("Checkpoint path", value="../src/chess_style_net.pt")
fen_input = st.sidebar.text_input("FEN (blank = starting position)", value="")

if not os.path.exists(ckpt_path):
    st.warning(f"No checkpoint found at `{ckpt_path}`. Train a model first (see README) then point this at the .pt file.")
    st.stop()

model, player_id_map = load_model(ckpt_path)
ids = list(player_id_map.values())
names = {v: k for k, v in player_id_map.items()}

if len(ids) < 2:
    st.error("Need at least 2 styles in the checkpoint to demo interpolation.")
    st.stop()

col_a, col_b = st.sidebar.columns(2)
style_a = col_a.selectbox("Style A", ids, format_func=lambda i: names[i], index=0)
style_b = col_b.selectbox("Style B", ids, format_func=lambda i: names[i], index=min(1, len(ids) - 1))
alpha = st.sidebar.slider(f"Blend (1.0 = pure {names[style_a]}, 0.0 = pure {names[style_b]})", 0.0, 1.0, 0.5, 0.05)

try:
    board = chess.Board(fen_input) if fen_input.strip() else chess.Board()
except ValueError:
    st.error("Invalid FEN, showing starting position instead.")
    board = chess.Board()

style_vec = model.interpolate_style(style_a, style_b, alpha)
moves, blunder_p = predict_top_moves(model, board, style_vec)

left, right = st.columns([1, 1])

with left:
    st.subheader("Position")
    svg = chess.svg.board(board, size=400)
    st.image(svg.encode(), use_column_width=False)

with right:
    st.subheader(f"Top predicted moves (alpha={alpha:.2f})")
    for mv, p in moves:
        st.write(f"**{mv.uci()}**  —  {p:.1%}")
        st.progress(min(p, 1.0))
    if blunder_p is not None:
        st.metric("Predicted blunder risk (aux head)", f"{blunder_p:.1%}")

st.divider()
st.caption(
    "Note: the blunder-risk head is only meaningful if trained with real "
    "Lichess %eval-derived blunder labels — see README for how to wire that up."
)
