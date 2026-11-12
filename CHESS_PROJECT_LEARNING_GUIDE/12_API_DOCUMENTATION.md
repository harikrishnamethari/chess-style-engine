# 12 — API Documentation & Route Specifications

This reference documents all REST and WebSocket endpoints in the backend.

---

## 1. REST Endpoints (`/api/v1/games`)

### 1. `POST /api/v1/games/upload`
- **Purpose**: Uploads a standard PGN file, extracts game metrics, calculates a 32d style embedding via the GRU `HistoryEncoder`, saves records to the database, and returns the profile.
- **Content-Type**: `multipart/form-data`
- **Request Parameters**:
  - `file`: `UploadFile` (binary/text PGN)
  - `username`: `str` (query param, default: `"anonymous"`)
- **Response Format (`200 OK`)**:
  ```json
  {
    "status": "ok",
    "game_id": 42,
    "metadata": {
      "white": "PlayerA",
      "black": "PlayerB",
      "result": "1-0",
      "num_moves": 38
    },
    "history_features": [[28.5, 0.14, 38], ...],
    "style_embedding_dim": 32,
    "profile": {
      "avg_cp_loss": 28.5,
      "capture_rate": 0.14,
      "games_analysed": 1
    }
  }
  ```
- **Error Codes**: `400 Bad Request` (Invalid UTF-8), `422 Unprocessable Entity` (Empty/invalid PGN).

---

### 2. `GET /api/v1/games/profile/{username}`
- **Purpose**: Retrieves a player's saved style profile and mathematical embedding vector.
- **Request Parameters**: `username` (path parameter)
- **Response Format (`200 OK`)**:
  ```json
  {
    "username": "hikaru",
    "avg_cp_loss": 18.2,
    "capture_rate": 0.165,
    "games_analysed": 15,
    "style_embedding": [0.124, -0.452, ..., 0.891]
  }
  ```
- **Error Codes**: `404 Not Found` (User does not exist or has not uploaded games).

---

### 3. `GET /api/v1/games/similarity`
- **Purpose**: Calculates the **Cosine Similarity** between two users' style embeddings:
  $$\text{Similarity}(A, B) = \frac{\mathbf{v}_A \cdot \mathbf{v}_B}{\|\mathbf{v}_A\|_2 \|\mathbf{v}_B\|_2}$$
- **Request Parameters**:
  - `username_a`: `str` (query param)
  - `username_b`: `str` (query param)
- **Response Format (`200 OK`)**:
  ```json
  {
    "username_a": "player1",
    "username_b": "player2",
    "cosine_similarity": 0.874,
    "style_description": "Very Similar Tactically"
  }
  ```

---

### 4. `GET /health`
- **Purpose**: Readiness probe and uptime monitor.
- **Response Format (`200 OK`)**:
  ```json
  {
    "status": "ok",
    "version": "2.0.0"
  }
  ```

---

## 2. WebSocket Protocol (`/ws/play`)

- **Protocol**: `ws://` / `wss://`
- **Client Frame 1: Set Style**
  ```json
  {
    "type": "set_style",
    "history": [[25.0, 0.12, 15.0], [30.0, 0.14, 22.0], ...]
  }
  ```
- **Client Frame 2: Evaluate Position**
  ```json
  {
    "type": "evaluate",
    "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
    "top_k": 5
  }
  ```
- **Server Response Frame**
  ```json
  {
    "type": "evaluation",
    "fen": "rnbqkbnr/...",
    "top_moves": [
      { "move": "c7c5", "prob": 0.384 },
      { "move": "e7e5", "prob": 0.291 },
      { "move": "e7e6", "prob": 0.142 }
    ],
    "blunder_risk": 0.045
  }
  ```

