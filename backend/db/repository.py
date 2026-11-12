"""
backend/db/repository.py
Repository Pattern — Interface Segregation: one focused class per entity.
Business logic in routes never touches SQLAlchemy directly.
"""
from sqlalchemy.orm import Session
from backend.db.models import User, Game, PlayerProfile


class UserRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_or_create(self, username: str) -> User:
        user = self.db.query(User).filter(User.username == username).first()
        if not user:
            user = User(username=username)
            self.db.add(user)
            self.db.commit()
            self.db.refresh(user)
        return user


class GameRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(self, user_id: int, pgn: str, white: str = None, black: str = None) -> Game:
        game = Game(user_id=user_id, pgn=pgn, white_player=white, black_player=black)
        self.db.add(game)
        self.db.commit()
        self.db.refresh(game)
        return game

    def list_for_user(self, user_id: int) -> list[Game]:
        return self.db.query(Game).filter(Game.user_id == user_id).all()


class ProfileRepository:
    def __init__(self, db: Session):
        self.db = db

    def upsert(self, user_id: int, **kwargs) -> PlayerProfile:
        profile = self.db.query(PlayerProfile).filter(PlayerProfile.user_id == user_id).first()
        if not profile:
            profile = PlayerProfile(user_id=user_id, **kwargs)
            self.db.add(profile)
        else:
            for k, v in kwargs.items():
                setattr(profile, k, v)
        self.db.commit()
        self.db.refresh(profile)
        return profile

    def get(self, user_id: int) -> PlayerProfile | None:
        return self.db.query(PlayerProfile).filter(PlayerProfile.user_id == user_id).first()
