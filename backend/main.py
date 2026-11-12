"""
backend/main.py  —  App factory only.
Open/Closed: add new routers here without touching any other module.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.core.config import settings
from backend.db.session import create_tables
from backend.api.routes import games, ws


def create_app() -> FastAPI:
    app = FastAPI(
        title="Chess Style Engine API",
        description="Style-conditioned chess AI — REST + WebSocket interface",
        version="2.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Create DB tables on startup
    @app.on_event("startup")
    def on_startup():
        create_tables()

    # Register routers
    app.include_router(games.router, prefix=settings.api_prefix)
    app.include_router(ws.router)

    @app.get("/health")
    def health():
        return {"status": "ok", "version": "2.0.0"}

    return app


app = create_app()
