"""
backend/core/config.py
Single Responsibility: all configuration in one place.
All other modules import from here — never hardcode connection strings.
"""
import os
from dataclasses import dataclass
from pathlib import Path

# Load .env if present
_env_path = Path(__file__).resolve().parent.parent.parent / ".env"
if _env_path.exists():
    with open(_env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


@dataclass(frozen=True)
class Settings:
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./chess_style.db")
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    model_path: str = os.getenv("MODEL_PATH", "checkpoints/my_style_net.pt")
    cors_origins: tuple = (
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "null",
        "*",
    )
    api_prefix: str = "/api/v1"
    hist_seq_len: int = 10
    hist_features: int = 3


settings = Settings()
