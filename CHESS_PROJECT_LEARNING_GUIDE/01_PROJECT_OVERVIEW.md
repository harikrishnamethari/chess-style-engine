# 01 — Project Overview: Chess Style Engine 🎭♟️

## 1. What is this project?
The **Chess Style Engine** is an end-to-end, full-stack Machine Learning application and AI engine. Unlike traditional chess engines (such as Stockfish) whose singular objective is to calculate the **theoretically optimal move**, Chess Style Engine aims to **model and reproduce human play styles**. 

It answers the question:
> *"Given this board state and a specific player's historical habits (tactical aggression, centipawn accuracy, capture rate), what move would this specific human play?"*

---

## 2. Core Pillars of the Application

```
                    ┌───────────────────────────────┐
                    │       Human Chess Games       │
                    │   (PGN / Lichess Game Dumps)  │
                    └──────────────┬────────────────┘
                                   │
                                   ▼
                    ┌───────────────────────────────┐
                    │      ML Pipeline & Model      │
                    │         ChessStyleNet         │
                    │  (ResNet + GRU Style Encoder) │
                    └──────────────┬────────────────┘
                                   │
                                   ▼
                    ┌───────────────────────────────┐
                    │        FastAPI Backend        │
                    │   REST Endpoints + WebSocket  │
                    │  (PostgreSQL + Redis Caching) │
                    └──────────────┬────────────────┘
                                   │
                                   ▼
                    ┌───────────────────────────────┐
                    │       Vanilla Frontend        │
                    │   HTML5 / CSS3 / JS (ES6+)    │
                    │  (Chessboard.js + Chart.js)   │
                    └───────────────────────────────┘
```

1. **Machine Learning Pipeline (`src/`)**:
   - AlphaZero-style 14-channel $8 \times 8$ board state representation.
   - 4,672-dimensional discrete move action space covering Queen moves, Knight moves, and underpromotions.
   - **`ChessStyleNet`**: A hybrid architecture consisting of an 8-residual-block CNN torso, an auxiliary GRU-based `HistoryEncoder` that outputs a 32-dimensional continuous style embedding vector $\mathbf{v} \in \mathbb{R}^{32}$, a policy head with legal move masking, and a multi-task blunder detection head.
2. **FastAPI Backend (`backend/`)**:
   - Clean architecture adhering to SOLID design principles.
   - Asynchronous endpoints (`backend/api/routes/games.py`) for PGN parsing, statistical extraction, and style profiling.
   - Full duplex WebSocket endpoint (`/ws/play`) for real-time move inference, blunder risk calculation, and live style adaptation.
   - Redis caching for inference evaluations to eliminate duplicate model forward passes.
   - PostgreSQL persistence using SQLAlchemy ORM for users, game records, and player style profiles.
3. **Frontend Client (`frontend/`)**:
   - 100% vanilla modern JavaScript (ES6+), HTML5, and CSS3 — zero dependencies on heavy frameworks like React, Next.js, or Vue.
   - Three dedicated pages:
     - `index.html`: Dashboard, PGN uploader, and player profile card.
     - `play.html`: Interactive board using `chessboard.js` + `chess.js` connected to the engine via WebSocket.
     - `similarity.html`: Visual style space and cosine similarity engine powered by `Chart.js`.
4. **Agentic Chess Coach (`agent/`)**:
   - An autonomous coaching module with long-term memory store (`memory_store.py`) and LLM integration (`coach_agent.py`) that analyzes past games to give natural language strategic advice.

---

## 3. High-Level User Journey

1. **Data Ingestion**: A user uploads their PGN history or fetches it directly from the Lichess API.
2. **Feature Extraction**: The backend parses the PGN into game-level statistics: average centipawn loss ($cp$), capture frequency ($cap$), and average game duration.
3. **Style Profiling**: The GRU `HistoryEncoder` compresses these temporal sequence metrics into a 32-dimensional style embedding.
4. **Interactive Play**: When playing in the browser, each position is dispatched over WebSocket. The backend encodes the FEN into a $(14, 8, 8)$ tensor, combines it with the style vector, runs a forward pass through `ChessStyleNet`, applies legal move masking, and returns the top-$k$ moves with probability scores and blunder risk.
5. **Style Blending**: Users can interpolate between two different player style vectors ($\alpha \mathbf{v}_A + (1-\alpha) \mathbf{v}_B$) to create synthetic hybrid playing personalities (e.g., 50% Aggressive Attacker + 50% Positional Grinder).

