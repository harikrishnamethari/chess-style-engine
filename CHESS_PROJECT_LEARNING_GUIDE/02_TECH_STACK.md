# 02 — Complete Tech Stack & Evidence

This document maps every single technology, framework, and library used in the codebase directly to the exact file and lines proving its usage.

---

## 1. Backend & Web Framework

| Technology | Purpose | Actual File Proof | Verification Details |
|---|---|---|---|
| **Python 3.10+** | Programming Language | `requirements.txt`, syntax in `src/ml/model.py` | Type hinting (`list[float]`, `dict[str, Any]`), PyTorch 2.x support |
| **FastAPI** | REST & WebSocket Web Framework | `backend/main.py:5`, `requirements.txt:9` | `from fastapi import FastAPI, APIRouter, WebSocket` |
| **Uvicorn** | ASGI Web Server | `requirements.txt:10`, `docker-compose.yml:25` | `uvicorn>=0.23`, runs `uvicorn backend.main:app --host 0.0.0.0 --port 8000` |
| **Pydantic / Pydantic Settings** | Data Validation & Env Config | `backend/core/config.py:7` | `from pydantic_settings import BaseSettings` |
| **WebSockets** | Real-time bidirectional streaming | `backend/api/routes/ws.py:7`, `requirements.txt:12` | `@router.websocket("/ws/play")` |
| **Starlette CORS Middleware** | Cross-Origin Resource Sharing | `backend/main.py:6` | `from fastapi.middleware.cors import CORSMiddleware` |

---

## 2. Database & Caching Layer

| Technology | Purpose | Actual File Proof | Verification Details |
|---|---|---|---|
| **PostgreSQL** | Primary Relational Database | `docker-compose.yml:4`, `backend/core/config.py:10` | Image: `postgres:15-alpine`, `database_url: str = "postgresql://..."` |
| **SQLAlchemy 2.0** | Object-Relational Mapping (ORM) | `backend/db/models.py:5-9`, `requirements.txt:14` | `from sqlalchemy.orm import relationship, DeclarativeBase` |
| **Redis** | In-memory Cache for ML Inference | `backend/ml/inference.py:13,46`, `docker-compose.yml:13` | Image: `redis:7-alpine`, `self.cache = redis.Redis.from_url(...)` |

---

## 3. Machine Learning & Chess Libraries

| Technology | Purpose | Actual File Proof | Verification Details |
|---|---|---|---|
| **PyTorch (torch)** | Deep Learning Framework | `src/ml/model.py:11`, `requirements.txt:2` | `import torch; import torch.nn as nn; torch>=2.0` |
| **python-chess** | Board logic, FEN, PGN parsing, move gen | `src/core/board_encoding.py:11`, `requirements.txt:3` | `import chess; import chess.pgn` |
| **NumPy** | Multidimensional array operations | `src/core/board_encoding.py:10`, `requirements.txt:4` | `import numpy as np; np.zeros((14, 8, 8))` |
| **Zstandard (zstd)** | Decompression for Lichess monthly dumps | `data/download_lichess.py:37`, `requirements.txt:5` | `import zstandard as zstd` |
| **Pytest** | Automated Testing Framework | `tests/test_board_encoding.py:11`, `requirements.txt:23` | `pytest>=7.4` |

---

## 4. Frontend Architecture & Libraries

| Technology | Purpose | Actual File Proof | Verification Details |
|---|---|---|---|
| **HTML5 & Vanilla JavaScript** | UI Layer & Client Logic | `frontend/index.html`, `frontend/play.html`, `frontend/js/ui.js` | Zero npm build steps; uses modern ES6 `fetch()`, `async/await`, `WebSocket` |
| **Custom CSS3** | Styling & Dark Theme Design | `frontend/style.css` | Flexbox, CSS Grid, custom color variables (`--bg-primary: #121418`) |
| **Chessboard.js** | Interactive graphical chessboard UI | `frontend/play.html:12,18` | Loaded via cdnjs: `chessboard-1.0.0.min.js` + `chessboard-1.0.0.min.css` |
| **Chess.js** | Client-side move validation & FEN tracking | `frontend/play.html:19` | Loaded via cdnjs: `chess.js@0.13.4` |
| **Chart.js** | 2D Style Embedding Scatter Plot | `frontend/similarity.html:12` | Loaded via CDN: `chart.js@4.4.0` |

---

## 5. Deployment & Containerization

| Technology | Purpose | Actual File Proof | Verification Details |
|---|---|---|---|
| **Docker Compose** | Multi-container orchestration | `docker-compose.yml:1` | Services: `postgres`, `redis`, `api` |
| **Google Colab** | Cloud GPU Training Environment | `notebooks/ChessStyleEngine_Colab.ipynb` | Jupyter notebook configured for free T4 GPU training |

