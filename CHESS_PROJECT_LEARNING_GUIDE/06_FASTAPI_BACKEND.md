# 06 — FastAPI Backend Architecture & Implementation

## 1. How the Application Boots

The backend initialization follows the **Application Factory Pattern** inside `backend/main.py`:

```python
def create_app() -> FastAPI:
    app = FastAPI(
        title="Chess Style Engine API",
        description="Style-conditioned chess AI — REST + WebSocket interface",
        version="2.0.0",
    )
    # 1. Attach Cross-Origin Resource Sharing (CORS)
    app.add_middleware(CORSMiddleware, ...)
    
    # 2. Database table initialization on startup
    @app.on_event("startup")
    def on_startup():
        create_tables()

    # 3. Router inclusion
    app.include_router(games.router, prefix=settings.api_prefix)
    app.include_router(ws.router)

    return app

app = create_app()
```

When you execute:
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```
Uvicorn imports the `app` instance, runs the `@app.on_event("startup")` hook to ensure PostgreSQL tables (`users`, `player_profiles`, `games`) exist, and exposes the HTTP & WebSocket listeners.

---

## 2. Configuration & Environment (`backend/core/config.py`)

Configuration is managed via **Pydantic Settings**:
```python
class Settings(BaseSettings):
    database_url: str = "postgresql://user:password@localhost:5432/chess_style_db"
    redis_url: str = "redis://localhost:6379"
    model_path: str = "checkpoints/chess_style_net.pt"
    api_prefix: str = "/api/v1"
    cors_origins: list[str] = ["*"]
    hist_seq_len: int = 10
    hist_num_features: int = 3
```
- It automatically reads overrides from `.env` or system environment variables without hardcoding secrets in source code.

---

## 3. Dependency Injection & Repository Pattern

Instead of embedding raw SQL queries or ORM calls directly in route functions, the project adheres to the **Single Responsibility Principle (SRP)** and **Dependency Inversion Principle (DIP)**:

1. **Database Session (`backend/db/session.py`)**:
   ```python
   def get_db():
       db = SessionLocal()
       try:
           yield db
       finally:
           db.close()
   ```
   FastAPI's `Depends(get_db)` injects an active SQLAlchemy session per request and guarantees cleanup when the request finishes.

2. **Repository Layer (`backend/db/repository.py`)**:
   Encapsulates all database mutations:
   - `UserRepository`: `get_or_create(username)`
   - `GameRepository`: `create(user_id, pgn, white, black)`
   - `ProfileRepository`: `upsert(user_id, style_embedding, avg_cp_loss, ...)`

---

## 4. WebSocket Streaming Protocol (`backend/api/routes/ws.py`)

Real-time move evaluation uses a persistent full-duplex WebSocket at `/ws/play`:

```mermaid
sequenceDiagram
    autonumber
    Client->>Server: Connect ws://localhost:8000/ws/play
    Server-->>Client: Accept Connection
    
    opt Set Style
        Client->>Server: {"type": "set_style", "history": [[25.0, 0.12, 15], ...]}
        Server-->>Client: {"type": "info", "message": "Style updated"}
    end

    loop Game Play
        Client->>Server: {"type": "evaluate", "fen": "rnbqkbnr/...", "top_k": 5}
        Server->>Server: InferenceEngine.get_top_moves(fen, history)
        Server-->>Client: {"type": "evaluation", "fen": "...", "top_moves": [...], "blunder_risk": 0.08}
    end
```

---

## 5. In-Memory Redis Caching (`backend/ml/inference.py`)

Neural network forward passes on a CPU/GPU are computationally non-trivial. When users revisit identical board positions with the same player history, `InferenceEngine` caches results in Redis:

- **Cache Key**: A deterministic SHA256 digest of `{ "fen": fen, "history": history }`.
- **Graceful Degradation**: If Redis is offline or not installed, `InferenceEngine` catches the connection exception and falls back to live model execution without crashing.

