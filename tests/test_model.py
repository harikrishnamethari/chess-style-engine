"""
tests/test_model.py
Unit tests for ChessStyleNet + HistoryEncoder forward pass.
Run: pytest tests/
"""
import sys
import os
import pytest

# Skip entire module if torch is not installed (e.g., still downloading)
torch = pytest.importorskip("torch", reason="torch not installed — skipping model tests")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ml.model import ChessStyleNet, HistoryEncoder
from src.core.board_encoding import NUM_MOVE_CLASSES


@pytest.fixture
def model():
    return ChessStyleNet()


@pytest.fixture
def batch():
    B = 4
    return {
        "board":   torch.randn(B, 18, 8, 8),
        "history": torch.randn(B, 10, 3),
    }


def test_policy_logits_shape(model, batch):
    logits, _ = model(batch["board"], history_tensor=batch["history"])
    assert logits.shape == (4, NUM_MOVE_CLASSES), \
        f"Expected (4, {NUM_MOVE_CLASSES}), got {logits.shape}"


def test_blunder_head_shape(model, batch):
    _, blunder = model(batch["board"], history_tensor=batch["history"])
    assert blunder is not None
    assert blunder.shape == (4, 1), f"Expected (4, 1), got {blunder.shape}"


def test_style_vector_from_history(model):
    hist = torch.randn(2, 10, 3)
    sv = model.style_vector(history_tensor=hist)
    assert sv.shape == (2, model.style_dim)


def test_style_vec_override(model, batch):
    """Passing style_vec directly should bypass history encoder."""
    sv = torch.randn(4, model.style_dim)
    logits, _ = model(batch["board"], style_vec=sv)
    assert logits.shape == (4, NUM_MOVE_CLASSES)


def test_style_interpolation(model):
    hist_a = torch.randn(1, 10, 3)
    hist_b = torch.randn(1, 10, 3)
    va = model.style_vector(history_tensor=hist_a)
    vb = model.style_vector(history_tensor=hist_b)
    blend = 0.5 * va + 0.5 * vb
    assert blend.shape == (1, model.style_dim)


def test_history_encoder_standalone():
    enc = HistoryEncoder(num_features=3, hidden_size=64, style_dim=32)
    x = torch.randn(8, 10, 3)
    out = enc(x)
    assert out.shape == (8, 32)


def test_no_nan_in_logits(model, batch):
    logits, blunder = model(batch["board"], history_tensor=batch["history"])
    assert not torch.isnan(logits).any(), "NaN in policy logits"
    assert not torch.isnan(blunder).any(), "NaN in blunder head"


def test_param_count(model):
    """Model should be compact enough for fast Colab training."""
    n = sum(p.numel() for p in model.parameters())
    assert n < 2_000_000, f"Model too large: {n:,} params"
    print(f"\nParam count: {n:,}")
