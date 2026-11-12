# 13 — Feature-by-Feature Deep Dive

This document breaks down every end-user feature in the system, tracing the exact chain of code files involved from browser to database.

---

## Feature 1: PGN Upload & Style Profile Extraction
- **Description**: Users upload their recorded chess games. The system parses the PGN, computes playing metrics, derives a 32d style embedding via the GRU subnetwork, and saves the profile.
- **Frontend File**: `frontend/index.html` (form) + `frontend/js/api.js` (`uploadPGN`) + `frontend/js/ui.js` (`renderProfileCard`).
- **FastAPI Route**: `POST /api/v1/games/upload` in `backend/api/routes/games.py`.
- **Parsing Logic**: `backend/ml/pgn_parser.py` (`extract_game_metadata`, `extract_history_features`).
- **ML Subnet**: `backend/ml/inference.py` calling `model.history_encoder`.
- **Database Tables**: `users`, `games`, `player_profiles` via `UserRepository`, `GameRepository`, `ProfileRepository`.
- **Manual Verification Step**:
  ```bash
  curl -X POST "http://localhost:8000/api/v1/games/upload?username=testuser" \
    -F "file=@agent/sample_games.pgn"
  ```

---

## Feature 2: Real-time Interactive Play with WebSocket Inference
- **Description**: Users play on an interactive chessboard. As moves are made, the position is evaluated in real time, returning style-conditioned top moves and blunder probability.
- **Frontend File**: `frontend/play.html` (with `chessboard.js` and `chess.js`).
- **WebSocket Endpoint**: `/ws/play` in `backend/api/routes/ws.py`.
- **Inference Engine**: `backend/ml/inference.py` (`InferenceEngine.get_top_moves`).
- **Core ML Transformation**: `src/core/board_encoding.py` (`board_to_tensor`, `legal_move_mask`).
- **Caching**: Redis caches the top-$k$ moves using key `cse:<sha256>`.
- **Manual Verification Step**:
  Open `frontend/play.html` in browser, open browser DevTools Network > WS tab, drag any piece, and inspect incoming JSON evaluation payloads.

---

## Feature 3: Style Similarity Comparison
- **Description**: Calculates mathematical alignment between two players' playing styles using cosine similarity on their GRU embedding vectors.
- **Frontend File**: `frontend/similarity.html` (with `Chart.js`).
- **FastAPI Route**: `GET /api/v1/games/similarity` in `backend/api/routes/games.py`.
- **Formula**:
  $$\text{sim}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$
- **Manual Verification Step**:
  ```bash
  curl "http://localhost:8000/api/v1/games/similarity?username_a=hikaru&username_b=magnus"
  ```

---

## Feature 4: Style Interpolation (Synthetic Hybrid Styles)
- **Description**: Blends two distinct player styles in arbitrary proportions (e.g., $70\%$ Player A $+ 30\%$ Player B) by taking a convex linear combination of their embedding vectors:
  $$\mathbf{v}_{\text{blend}} = \alpha \mathbf{v}_A + (1 - \alpha) \mathbf{v}_B$$
- **Code Reference**: `src/interpolate_demo.py` and `ChessStyleNet.forward(..., style_vec=...)`.
- **Verification**: Verified in `tests/test_model.py:test_style_interpolation` which asserts that overriding `style_vec` changes the policy logits deterministically without crashing.

