# 17 — 10-Day Hands-on Learning & Mastery Plan

This structured 10-day roadmap takes you from initial understanding to total mastery of the codebase, ensuring you can explain every detail with confidence.

---

### Day 1: Chess Representation & Board Encoding
- **Concepts**: Bitboards, FEN notation, 14-channel spatial tensors, 4,672-action space.
- **Files to Inspect**: `src/core/board_encoding.py`, `tests/test_board_encoding.py`.
- **Key Functions**: `board_to_tensor()`, `move_to_index()`, `index_to_move()`, `legal_move_mask()`.
- **Coding Exercise**: Write a small script that takes the starting chess board, pushes `1. e4`, converts it to a tensor, and prints `tensor.shape` and active planes.
- **Check Question**: Why do we flip the board representation vertically when it's Black's turn?

---

### Day 2: PyTorch Neural Architecture (ChessStyleNet)
- **Concepts**: Residual blocks, Convolutional networks for game boards, Multi-task heads.
- **Files to Inspect**: `src/ml/model.py`, `tests/test_model.py`.
- **Key Classes**: `ResBlock`, `ChessStyleNet`.
- **Coding Exercise**: Instantiate `ChessStyleNet`, pass a dummy tensor of shape `(2, 14, 8, 8)` with `torch.randn`, and verify that policy logits have shape `(2, 4672)` and blunder risk has shape `(2, 1)`.
- **Check Question**: What is the mathematical formula of a Residual Block with identity skip connections?

---

### Day 3: Temporal Style Modeling with GRU
- **Concepts**: History sequences, Recurrent Neural Networks, Latent style vectors, Style interpolation.
- **Files to Inspect**: `src/ml/model.py` (`HistoryEncoder`), `src/interpolate_demo.py`.
- **Key Functions**: `HistoryEncoder.forward()`, `ChessStyleNet.forward(style_vec=...)`.
- **Coding Exercise**: Generate two random style vectors $\mathbf{v}_1, \mathbf{v}_2 \in \mathbb{R}^{1 \times 32}$. Compute $\mathbf{v}_{\text{blend}} = 0.5 \mathbf{v}_1 + 0.5 \mathbf{v}_2$. Run the forward pass with both vectors and check how top move probabilities change.
- **Check Question**: Why is a continuous GRU style embedding superior to a one-hot `nn.Embedding(player_id)`?

---

### Day 4: Data Ingestion & PGN Processing
- **Concepts**: PGN parsing, Centipawn loss, Capture rate, Mainline move iteration.
- **Files to Inspect**: `backend/ml/pgn_parser.py`, `data/generate_sample_data.py`.
- **Key Functions**: `extract_game_metadata()`, `extract_history_features()`.
- **Coding Exercise**: Run `python data/generate_sample_data.py data/test.csv --num_games 10` and inspect the generated CSV rows.
- **Check Question**: How does the parser distinguish between an empty/invalid PGN and a valid short game?

---

### Day 5: Model Training Loop & Multi-Task Loss
- **Concepts**: Cross-entropy, Binary cross-entropy, Learning rate scheduling, Checkpointing.
- **Files to Inspect**: `src/training/trainer.py`, `src/training/train.py`, `src/utils/checkpoint.py`.
- **Key Classes**: `ChessDataset`, `Trainer`.
- **Coding Exercise**: Run a 1-epoch quick training test:
  ```bash
  python src/training/train.py --csv data/sample_data.csv --epochs 1 --batch_size 16
  ```
- **Check Question**: What is the purpose of gradient clipping at `max_norm=1.0`?

---

### Day 6: Evaluation Metrics & Style Consistency
- **Concepts**: Top-1 vs Top-k accuracy, Legal move compliance, KL divergence.
- **Files to Inspect**: `src/evaluation/metrics.py`, `src/evaluation/evaluate.py`.
- **Key Functions**: `top_k_accuracy()`, `legal_move_accuracy()`, `style_consistency_kl()`.
- **Coding Exercise**: Run `python src/evaluation/evaluate.py --csv data/sample_data.csv` and analyze the output metrics.
- **Check Question**: What does a high KL divergence indicate when comparing model outputs across two different player style vectors?

---

### Day 7: FastAPI Backend & SOLID Architecture
- **Concepts**: Application Factory, Dependency Injection, Pydantic Settings, Repository Pattern.
- **Files to Inspect**: `backend/main.py`, `backend/core/config.py`, `backend/db/session.py`, `backend/db/repository.py`.
- **Key Functions**: `create_app()`, `get_db()`, `UserRepository.get_or_create()`.
- **Coding Exercise**: Start the API locally (`uvicorn backend.main:app --reload`) and visit `http://localhost:8000/docs` to test endpoints interactively via Swagger UI.
- **Check Question**: Why should database queries be placed inside Repository classes instead of route handler functions?

---

### Day 8: WebSocket Inference & Redis Caching
- **Concepts**: Full-duplex WebSocket communication, SHA256 cache keys, In-memory caching.
- **Files to Inspect**: `backend/api/routes/ws.py`, `backend/ml/inference.py`.
- **Key Classes**: `InferenceEngine`, `play_websocket`.
- **Coding Exercise**: Add a `print("[Cache Hit!]")` log inside `InferenceEngine.get_top_moves()` and verify that querying the same FEN twice hits the cache.
- **Check Question**: How does the system handle inference when Redis is offline?

---

### Day 9: Frontend Architecture & Client-Side Logic
- **Concepts**: Vanilla JS (ES6+), Fetch API, DOM manipulation, `chessboard.js`, `chess.js`, `Chart.js`.
- **Files to Inspect**: `frontend/index.html`, `frontend/play.html`, `frontend/similarity.html`, `frontend/js/api.js`, `frontend/js/ui.js`.
- **Coding Exercise**: Open `frontend/play.html` in your browser via `python -m http.server 5500 --directory frontend` and test piece movement.
- **Check Question**: What is the difference in responsibility between `chessboard.js` and `chess.js`?

---

### Day 10: Mock Technical Interview Simulation
- **Goal**: Rehearse your interview presentation out loud without referencing your notes.
- **Tasks**:
  1. Deliver the 30-second elevator pitch cleanly.
  2. Deliver the 2-minute architectural explanation.
  3. Answer 5 random questions from [16_INTERVIEW_PREPARATION.md](file:///home/hari/Projects/chess-style-engine/CHESS_PROJECT_LEARNING_GUIDE/16_INTERVIEW_PREPARATION.md) from memory.
  4. Sketch the end-to-end dataflow diagram from a user's browser drag to the PyTorch forward pass on a whiteboard or piece of paper.

