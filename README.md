# Chess Style Engine 🎭♟️

A **full-stack ML system** that learns to play chess *like a specific person* — not optimally — by conditioning a ResNet+GRU policy network on a player's history of game statistics. Two styles can be interpolated at inference time.

> **Interview one-liner:** "I built a style-conditioned chess AI that encodes a player's last N games into a style embedding via a GRU, then fuses it with a ResNet board encoder to produce personalised move distributions — with a multi-task blunder-prediction head and a FastAPI + WebSocket backend served to a pure-HTML frontend."

---

## Architecture

```
Lichess PGN(s)
      │
      ▼
  PGN Parser                        backend/ml/pgn_parser.py
  (CP loss proxy, capture rate,
   castling timing)
      │
      ▼  (seq_len=10, features=3)
  History Encoder (GRU → Linear)    src/ml/model.py :: HistoryEncoder
      │
      ▼  (style_dim=32)
  Style Embedding  ──────────────────────────────────┐
                                                     │
  Board (FEN)                                        │
      │                                              │
      ▼                                              │
  Board Encoder (ResNet × 6 blocks) ─────────────────┤
                                                     ▼
                                              concat + project
                                                     │
                               ┌─────────────────────┤
                               │                     │
                               ▼                     ▼
                         Policy Head           Blunder Head
                         (4672 classes)        (BCE sigmoid)
                               │
                         Legal move masking
                               │
                         Top-k moves
```

---

## Project Layout

```
chess-style-engine/
├── src/                         # ML source code (SOLID-structured)
│   ├── core/
│   │   └── board_encoding.py    # AlphaZero 8×8×73 move encoding
│   ├── ml/
│   │   ├── model.py             # ChessStyleNet + HistoryEncoder
│   │   └── dataset.py           # PyTorch Dataset (CSV + stats.json)
│   ├── training/
│   │   ├── trainer.py           # Training loop class
│   │   └── train.py             # CLI entrypoint
│   ├── evaluation/
│   │   ├── metrics.py           # top_k_accuracy, style_consistency (KL)
│   │   └── evaluate.py          # CLI entrypoint
│   ├── utils/
│   │   └── checkpoint.py        # save/load model (DRY)
│   └── inference_demo.py        # CLI style-blend demo
│
├── backend/                     # FastAPI service (SOLID-structured)
│   ├── core/
│   │   └── config.py            # Centralised settings (env vars)
│   ├── db/
│   │   ├── models.py            # SQLAlchemy ORM models
│   │   ├── session.py           # DB session factory
│   │   └── repository.py        # Repository pattern CRUD
│   ├── ml/
│   │   ├── inference.py         # InferenceEngine singleton + Redis cache
│   │   └── pgn_parser.py        # PGN → history features
│   ├── api/
│   │   └── routes/
│   │       ├── games.py         # REST: /upload, /profile, /similarity
│   │       └── ws.py            # WebSocket: /ws/play
│   └── main.py                  # App factory
│
├── frontend/                    # Pure HTML/CSS/JS (no frameworks)
│   ├── index.html               # Dashboard — PGN upload + style profile
│   ├── play.html                # Interactive chessboard + WS engine
│   ├── similarity.html          # 2D style embedding scatter map
│   ├── style.css                # Shared dark glassmorphism design system
│   └── js/
│       ├── api.js               # All HTTP + WebSocket calls
│       └── ui.js                # Toast, loading states, DOM helpers
│
├── data/
│   ├── generate_sample_data.py  # Synthetic 2-style dataset (CPU testable)
│   └── download_lichess.py      # Real Lichess pipeline (run on Colab)
│
├── tests/
│   ├── test_board_encoding.py   # Move encoding collision + shape tests
│   ├── test_model.py            # Forward pass, shapes, NaN checks
│   └── test_pgn_parser.py       # PGN feature extraction tests
│
├── notebooks/
│   └── ChessStyleEngine_Colab.ipynb  # End-to-end GPU training
├── demo/
│   └── app.py                   # Original Streamlit demo (legacy)
├── docker-compose.yml           # PostgreSQL + Redis
├── .env.example                 # Environment variable template
└── requirements.txt             # All dependencies
```

---

## Quickstart

### 1. Prerequisites

```bash
# Postgres + Redis via Docker
docker-compose up -d

# Install all dependencies
pip install -r requirements.txt
cp .env.example .env    # edit DATABASE_URL / REDIS_URL / MODEL_PATH if needed
```

### 2. Generate sample data & train (CPU, ~2 min)

```bash
python data/generate_sample_data.py data/sample_data.csv
python src/training/train.py --csv data/sample_data.csv --epochs 8
```

### 3. Run tests

```bash
pytest tests/ -v
```

### 4. Start the backend

```bash
uvicorn backend.main:app --reload
# API docs: http://localhost:8000/docs
```

### 5. Open the frontend

```bash
# Any static file server, e.g.:
python -m http.server 5500 --directory frontend
# Then visit: http://localhost:5500
```

### 6. Real training on Colab

Open `notebooks/ChessStyleEngine_Colab.ipynb`, change runtime to GPU (T4), run all cells. Start with `--max_games 5000` then scale to 50k+.

---

## Key Technical Concepts (for interviews)

| Concept | Where |
|---|---|
| Conditional imitation learning | `ChessStyleNet.forward()` |
| GRU sequence encoder | `HistoryEncoder` — generalises to unseen players |
| Multi-task learning | Policy CE loss + Blunder BCE loss |
| AlphaZero move encoding | `src/core/board_encoding.py` — 8×8×73 planes |
| Custom KL metric | `src/evaluation/metrics.py :: style_consistency()` |
| Repository pattern | `backend/db/repository.py` |
| WebSocket streaming | `backend/api/routes/ws.py` |
| Redis caching | `backend/ml/inference.py` — 5-min TTL per (fen, history) |
| SOLID architecture | Throughout — single-file single-responsibility, no circular deps |

---

## SOLID Principles Applied

- **S**ingle Responsibility — each module does one thing (`pgn_parser.py` only parses, `checkpoint.py` only saves/loads, `metrics.py` only computes metrics)
- **O**pen/Closed — `Trainer` can have new loss terms added without touching `train.py`; new routes added to `main.py` without touching existing routes
- **L**iskov Substitution — `style_vec` can substitute `history_tensor` in `ChessStyleNet.forward()` transparently
- **I**nterface Segregation — `UserRepository`, `GameRepository`, `ProfileRepository` are separate; callers only depend on what they need
- **D**ependency Inversion — `InferenceEngine` depends on the abstract model interface, not on the training pipeline
