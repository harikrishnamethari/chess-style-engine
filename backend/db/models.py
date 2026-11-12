"""
backend/db/models.py
Single Responsibility: ORM model definitions only.
"""
from sqlalchemy import Column, Integer, String, Float, ForeignKey, JSON, DateTime, func
from sqlalchemy.orm import relationship, DeclarativeBase


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    created_at = Column(DateTime, server_default=func.now())

    profiles = relationship("PlayerProfile", back_populates="user", cascade="all, delete-orphan")
    games = relationship("Game", back_populates="user", cascade="all, delete-orphan")


class PlayerProfile(Base):
    __tablename__ = "player_profiles"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    style_embedding = Column(JSON, nullable=True)    # list[float] from HistoryEncoder
    avg_cp_loss = Column(Float, nullable=True)
    capture_rate = Column(Float, nullable=True)
    games_analysed = Column(Integer, default=0)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    user = relationship("User", back_populates="profiles")


class Game(Base):
    __tablename__ = "games"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    pgn = Column(String, nullable=False)
    white_player = Column(String, nullable=True)
    black_player = Column(String, nullable=True)
    uploaded_at = Column(DateTime, server_default=func.now())

    user = relationship("User", back_populates="games")
