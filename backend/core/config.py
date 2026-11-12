"""
backend/core/config.py
Single Responsibility: all configuration in one place.
All other modules import from here — never hardcode connection strings.
"""
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    database_url: str = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/chess_style_db")
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    model_path: str = os.getenv("MODEL_PATH", "chess_style_net.pt")
    cors_origins: list = ("http://localhost:5500", "http://127.0.0.1:5500", "null")
    api_prefix: str = "/api/v1"
    hist_seq_len: int = 10
    hist_features: int = 3


settings = Settings()
