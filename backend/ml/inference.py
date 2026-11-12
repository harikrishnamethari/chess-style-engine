"""
backend/ml/inference.py
Dependency Inversion: InferenceEngine depends on abstractions (model interface),
not on the concrete training setup. Loaded once as a singleton at startup.
Redis caching included for board evaluations.
"""
import sys
import os
import json
import hashlib
import torch
import chess
import redis

# Resolve src path
_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.core.board_encoding import board_to_tensor, move_to_index, legal_move_mask
from src.ml.model import ChessStyleNet
from backend.core.config import settings


class InferenceEngine:
    """
    Singleton wrapper around ChessStyleNet.
    Evaluates positions given a history sequence and returns top-k moves + blunder risk.
    Results are cached in Redis to avoid re-running the model for repeated positions.
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = ChessStyleNet()
        self.model.eval()
        self.model.to(self.device)

        if os.path.exists(settings.model_path):
            ckpt = torch.load(settings.model_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            print(f"[InferenceEngine] Loaded weights from {settings.model_path}")
        else:
            print(f"[InferenceEngine] WARNING: no checkpoint at '{settings.model_path}' — using random weights.")

        # Redis cache (graceful fallback if Redis unavailable)
        try:
            self.cache = redis.Redis.from_url(settings.redis_url, decode_responses=True)
            self.cache.ping()
            print("[InferenceEngine] Redis cache connected.")
        except Exception:
            self.cache = None
            print("[InferenceEngine] Redis unavailable — caching disabled.")

    def _cache_key(self, fen: str, history: list) -> str:
        payload = json.dumps({"fen": fen, "history": history}, sort_keys=True)
        return "cse:" + hashlib.sha256(payload.encode()).hexdigest()[:16]

    def get_top_moves(self, fen: str, history: list, top_k: int = 5) -> dict:
        """
        Args:
            fen:     Current board position in FEN notation.
            history: Shape (seq_len, num_features). Player's past game stats.
            top_k:   Number of top moves to return.
        Returns:
            {
              "top_moves": [{"move": "e2e4", "prob": 0.4}, ...],
              "blunder_risk": 0.12
            }
        """
        cache_key = self._cache_key(fen, history)
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached:
                return json.loads(cached)

        board = chess.Board(fen)
        board_tensor = torch.from_numpy(board_to_tensor(board)).unsqueeze(0).to(self.device)
        hist_tensor = torch.tensor([history], dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits, blunder_logit = self.model(board_tensor, history_tensor=hist_tensor)
            probs = torch.softmax(logits[0], dim=0)

        legal_moves = list(board.legal_moves)
        legal_probs = []
        for move in legal_moves:
            idx = move_to_index(move, board)
            legal_probs.append({"move": move.uci(), "prob": round(probs[idx].item(), 5)})

        legal_probs.sort(key=lambda x: x["prob"], reverse=True)
        blunder_risk = round(torch.sigmoid(blunder_logit[0]).item(), 4) if blunder_logit is not None else 0.0

        result = {"top_moves": legal_probs[:top_k], "blunder_risk": blunder_risk}

        if self.cache:
            self.cache.setex(cache_key, 300, json.dumps(result))  # 5-min TTL

        return result

    def get_style_embedding(self, history: list) -> list[float]:
        """Encode a history sequence into a style embedding vector."""
        hist_tensor = torch.tensor([history], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            style_vec = self.model.style_vector(history_tensor=hist_tensor)
        return style_vec.squeeze(0).tolist()

    def cosine_similarity(self, emb_a: list[float], emb_b: list[float]) -> float:
        a = torch.tensor(emb_a)
        b = torch.tensor(emb_b)
        return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


# Singleton — instantiated once at import time
engine = InferenceEngine()
