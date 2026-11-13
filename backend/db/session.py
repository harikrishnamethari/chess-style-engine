"""
backend/db/session.py
Single Responsibility: database session factory.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.core.config import settings
from backend.db.models import Base

engine_args = {}
if settings.database_url.startswith("sqlite"):
    engine_args["connect_args"] = {"check_same_thread": False}

engine = create_engine(settings.database_url, **engine_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def create_tables():
    """Call once at startup to create all tables."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency — yields a DB session, ensures it's closed."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
