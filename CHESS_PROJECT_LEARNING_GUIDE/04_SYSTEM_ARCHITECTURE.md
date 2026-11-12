# 04 — System Architecture Diagrams

This document contains Mermaid diagrams capturing the end-to-end system, request lifecycles, and cross-cutting concerns.

---

## Diagram 1: Complete End-to-End Application Architecture

```mermaid
flowchart TD
    subgraph Client ["Client Layer (Browser)"]
        UI_Dash["Dashboard (index.html)"]
        UI_Play["Interactive Play (play.html)"]
        UI_Sim["Style Map (similarity.html)"]
        JS_API["api.js (ES6 Fetch)"]
        JS_WS["WebSocket Client"]
    end

    subgraph Backend ["FastAPI Backend Layer (Uvicorn)"]
        App["FastAPI App Factory (main.py)"]
        CORS["CORS Middleware"]
        RouterGames["Games Router (/api/v1/games)"]
        RouterWS["WebSocket Router (/ws/play)"]
        PGNParser["PGN Parser (pgn_parser.py)"]
        Repo["Repository Layer (repository.py)"]
    end

    subgraph CachingAndDB ["Persistence & Caching"]
        Postgres[("PostgreSQL 15\n(Users, Profiles, Games)")]
        RedisCache[("Redis 7\n(FEN + Style Inference Cache)")]
    end

    subgraph ML ["Machine Learning System (Inference)"]
        InfEngine["InferenceEngine Singleton"]
        Model["ChessStyleNet\n(ResNet Torso + GRU HistoryEncoder)"]
        Weights[("Checkpoint File\n(chess_style_net.pt)")]
    end

    %% Client to Backend
    UI_Dash -->|Upload PGN / Fetch Profile| JS_API
    UI_Sim -->|Query Cosine Similarity| JS_API
    UI_Play -->|Real-time FEN stream| JS_WS

    JS_API -->|HTTP REST Requests| App
    JS_WS <-->|Bidirectional WS Events| App

    App --> CORS
    CORS --> RouterGames
    CORS --> RouterWS

    %% Games Router flow
    RouterGames -->|Parse PGN| PGNParser
    RouterGames -->|Store / Query data| Repo
    Repo -->|SQLAlchemy ORM| Postgres

    %% WebSocket flow & ML integration
    RouterWS -->|Query Top Moves & Blunder| InfEngine
    RouterGames -->|Compute Style Embedding| InfEngine

    InfEngine <-->|Check Cache (SHA256 FEN+Hist)| RedisCache
    InfEngine -->|Model Forward Pass| Model
    Weights -.->|Loaded on Startup| Model
    InfEngine -->|Return Legal Masked Top-K| RouterWS
    RouterWS -->|Stream Predictions| JS_WS
```

---

## Diagram 3: FastAPI Backend Request Lifecycle

```mermaid
sequenceDiagram
    autonumber
    actor User as Client (Browser)
    participant ASGI as Uvicorn ASGI Server
    participant App as FastAPI App
    participant Middleware as CORS Middleware
    participant Route as games.py (/upload)
    participant Parser as pgn_parser.py
    participant ML as InferenceEngine
    participant Repo as ProfileRepository
    participant DB as PostgreSQL Database

    User->>ASGI: POST /api/v1/games/upload (Multipart PGN)
    ASGI->>App: Forward Request
    App->>Middleware: Intercept & Validate Headers
    Middleware->>Route: Route Dispatch (Dependency Injection)
    
    Route->>Route: Validate File Encoding (UTF-8)
    Route->>Parser: extract_game_metadata(pgn_text)
    Parser-->>Route: Return {white, black, result, moves}
    
    Route->>Parser: extract_history_features(pgn_text, max_games=10)
    Parser-->>Route: Return Sequence [[cp, cap, moves], ...]
    
    Route->>ML: get_style_embedding(history)
    ML->>ML: GRU Forward Pass -> 32d vector
    ML-->>Route: Return style_embedding list[float]
    
    Route->>Repo: upsert(user_id, embedding, stats)
    Repo->>DB: INSERT / UPDATE player_profiles
    DB-->>Repo: Confirm Row Saved
    
    Route-->>User: 200 OK JSON {status: "ok", profile: {...}}
```

---

## Diagram 8: Database Entity-Relationship (ER) Diagram

```mermaid
erDiagram
    User ||--o{ Game : "uploads"
    User ||--o{ PlayerProfile : "has"

    User {
        int id PK "Serial, Primary Key"
        string username "Unique, Indexed, Not Null"
        datetime created_at "Default: func.now()"
    }

    Game {
        int id PK "Serial, Primary Key"
        int user_id FK "References users.id (CASCADE)"
        string pgn "Raw PGN Text, Not Null"
        string white_player "Nullable"
        string black_player "Nullable"
        datetime uploaded_at "Default: func.now()"
    }

    PlayerProfile {
        int id PK "Serial, Primary Key"
        int user_id FK "References users.id (CASCADE)"
        json style_embedding "32-dimensional float list from GRU"
        float avg_cp_loss "Average Centipawn Loss"
        float capture_rate "Captures per Move Rate"
        int games_analysed "Count of games processed"
        datetime updated_at "OnUpdate: func.now()"
    }
```

