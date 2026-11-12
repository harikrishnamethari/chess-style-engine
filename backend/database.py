# Superseded by backend/db/models.py + backend/db/session.py (SOLID restructure).
# Kept as stub to avoid import errors from old references.
from backend.db.models import User, Game, PlayerProfile, Base  # noqa: F401
from backend.db.session import SessionLocal, engine              # noqa: F401
