# 03 — Project Folder Structure & Module Breakdown

This document details the complete physical directory hierarchy, classifying every single file into **Essential**, **Supporting/Optional**, or **Generated/Legacy**.

```
chess-style-engine/
├── .env.example                       # [Essential] Environment variable template
├── .gitignore                          # [Essential] Git ignore rules for ML/PyTorch/Python
├── docker-compose.yml                  # [Essential] Orchestration for Postgres, Redis, and API
├── requirements.txt                    # [Essential] Unified Python dependencies
├── README.md                           # [Essential] Documentation, architecture & SOLID overview
│
├── backend/                            # FastAPI Backend Service (REST + WebSocket)
│   ├── main.py                         # [Essential] App factory, CORS setup, router registrations
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py                   # [Essential] Pydantic Settings (DB URLs, Redis, model path)
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── games.py                # [Essential] REST endpoints: /games/upload, /profile, /similarity
│   │       └── ws.py                   # [Essential] WebSocket endpoint: /ws/play (real-time inference)
│   ├── db/
│   │   ├── __init__.py
│   │   ├── models.py                   # [Essential] SQLAlchemy models: User, PlayerProfile, Game
│   │   ├── session.py                  # [Essential] Engine, SessionLocal, get_db dependency, create_tables
│   │   └── repository.py               # [Essential] Repository layer (UserRepository, GameRepository, ProfileRepository)
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── inference.py                # [Essential] Singleton InferenceEngine with Redis caching
│   │   └── pgn_parser.py               # [Essential] PGN metadata & sequential history feature extractor
│   ├── database.py                     # [Legacy Stub] Redirects to backend/db/ (prevents breaking old imports)
│   └── inference.py                    # [Legacy Stub] Redirects to backend/ml/inference.py
│
├── frontend/                           # Client-side UI (HTML5, Vanilla JS, CSS3)
│   ├── index.html                      # [Essential] Main dashboard & PGN file upload interface
│   ├── play.html                       # [Essential] Interactive play page (chessboard.js + WebSocket)
│   ├── similarity.html                 # [Essential] Style embedding visualization (Chart.js scatter plot)
│   ├── style.css                       # [Essential] Global stylesheet (dark modern dashboard theme)
│   └── js/
│       ├── api.js                      # [Essential] ES6 Fetch wrappers for all backend REST endpoints
│       └── ui.js                       # [Essential] DOM manipulation, alert banners, table updates
│
├── src/                                # Core Machine Learning Pipeline
│   ├── core/
│   │   ├── __init__.py
│   │   └── board_encoding.py           # [Essential] 14x8x8 board tensor & 4672 move action space encodings
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── model.py                    # [Essential] ChessStyleNet (ResNet + GRU HistoryEncoder + Heads)
│   │   └── dataset.py                  # [Essential] PyTorch Dataset (ChessDataset) for CSV/tensors
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train.py                    # [Essential] CLI entry point for training
│   │   └── trainer.py                  # [Essential] Trainer class (multi-task loss, validation, checkpoints)
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluate.py                 # [Essential] CLI evaluation entry point
│   │   └── metrics.py                  # [Essential] Top-k accuracy, blunder MAE, style consistency (KL)
│   ├── utils/
│   │   ├── __init__.py
│   │   └── checkpoint.py               # [Essential] Model weight saving and loading helpers
│   ├── board_encoding.py               # [Legacy Stub] Redirects to src.core.board_encoding
│   ├── model.py                        # [Legacy Stub] Redirects to src.ml.model
│   ├── dataset.py                      # [Legacy Stub] Redirects to src.ml.dataset
│   ├── train.py                        # [Legacy Stub] Redirects to src.training.train
│   ├── evaluate.py                     # [Legacy Stub] Redirects to src.evaluation.evaluate
│   ├── inference_demo.py               # [Optional] CLI move prediction demo on test positions
│   └── interpolate_demo.py             # [Optional] CLI style vector interpolation demo
│
├── data/                               # Data Generation & Ingestion
│   ├── download_lichess.py             # [Supporting] Streams & filters bulk Lichess monthly dumps (.zst)
│   ├── generate_sample_data.py         # [Supporting] Generates realistic synthetic games for offline testing
│   └── sample_data.csv                 # [Supporting] Ready-to-train sample dataset (1,000+ positions)
│
├── agent/                              # Agentic AI Chess Coach (LLM & Memory)
│   ├── coach_agent.py                  # [Supporting] LangChain/LLM Agent with tool calling
│   ├── memory_store.py                 # [Supporting] Persistent vector/JSON memory for student strengths/weaknesses
│   ├── tools.py                        # [Supporting] Tool interfaces for game evaluation & blunder analysis
│   ├── run_coach.py                    # [Supporting] Interactive CLI agent runner
│   ├── sample_games.pgn                # [Supporting] Sample PGN history for coach testing
│   └── requirements.txt                # [Supporting] Agent-specific dependencies
│
├── notebooks/
│   └── ChessStyleEngine_Colab.ipynb    # [Supporting] End-to-end Google Colab GPU training notebook
│
├── tests/                              # Automated Unit & Integration Tests
│   ├── __init__.py
│   ├── test_board_encoding.py          # [Essential] Verifies 14-plane shapes, move index bounds, reversibility
│   ├── test_model.py                   # [Essential] Verifies forward pass, tensor shapes, gradient flow, interpolation
│   └── test_pgn_parser.py              # [Essential] Verifies metadata extraction and history feature dimensions
│
└── demo/
    └── app.py                          # [Legacy] Early Streamlit demo prototype (superseded by frontend/)
```

---

## File Classification Summary

1. **Essential (Must have for production & interviews)**:
   - All modules in `backend/` (`main.py`, `routes/`, `db/`, `ml/`).
   - All modules in `frontend/` (`index.html`, `play.html`, `similarity.html`, `js/`, `style.css`).
   - All modules in `src/core/`, `src/ml/`, `src/training/`, `src/evaluation/`.
   - All modules in `tests/`.
   - Infrastructure: `docker-compose.yml`, `requirements.txt`.
2. **Supporting & Educational**:
   - `notebooks/ChessStyleEngine_Colab.ipynb`: Enables free GPU training.
   - `data/generate_sample_data.py`: Allows immediate local training without network downloads.
   - `agent/`: Demonstrates advanced AI capabilities (LLM tool-calling + memory).
3. **Legacy Stubs**:
   - Flat files directly in `src/` (`src/model.py`, `src/board_encoding.py`, etc.) and `backend/database.py` were converted to 1-line re-export stubs when refactoring to strict SOLID architecture, ensuring backward compatibility with older scripts.

