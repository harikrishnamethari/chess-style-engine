"""
backend/api/routes/games.py
Single Responsibility: HTTP endpoints for game/PGN management.
No DB/ML details — delegates to repositories and the inference engine.
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session

from backend.db.session import get_db
from backend.db.repository import UserRepository, GameRepository, ProfileRepository
from backend.ml.pgn_parser import extract_game_metadata, extract_history_features
from backend.ml.inference import engine as inference_engine

router = APIRouter(prefix="/games", tags=["games"])


@router.post("/upload")
async def upload_pgn(
    file: UploadFile = File(...),
    username: str = "anonymous",
    db: Session = Depends(get_db),
):
    """
    Upload a PGN file. Extracts player stats, computes style embedding,
    saves to DB, returns the profile.
    """
    content = await file.read()
    try:
        pgn_text = content.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be UTF-8 encoded text.")

    metadata = extract_game_metadata(pgn_text)
    if not metadata:
        raise HTTPException(status_code=422, detail="Could not parse PGN — check format.")

    history = extract_history_features(pgn_text, max_games=10)
    style_embedding = inference_engine.get_style_embedding(history)

    user_repo = UserRepository(db)
    game_repo = GameRepository(db)
    profile_repo = ProfileRepository(db)

    user = user_repo.get_or_create(username)
    game = game_repo.create(
        user_id=user.id,
        pgn=pgn_text,
        white=metadata["white"],
        black=metadata["black"],
    )
    avg_cp = sum(h[0] for h in history) / len(history)
    cap_rate = sum(h[1] for h in history) / len(history)
    profile = profile_repo.upsert(
        user_id=user.id,
        style_embedding=style_embedding,
        avg_cp_loss=avg_cp,
        capture_rate=cap_rate,
        games_analysed=(profile_repo.get(user.id).games_analysed or 0) + 1
        if profile_repo.get(user.id) else 1,
    )

    return {
        "status": "ok",
        "game_id": game.id,
        "metadata": metadata,
        "history_features": history,
        "style_embedding_dim": len(style_embedding),
        "profile": {
            "avg_cp_loss": round(avg_cp, 2),
            "capture_rate": round(cap_rate, 4),
            "games_analysed": profile.games_analysed,
        },
    }


@router.get("/profile/{username}")
def get_profile(username: str, db: Session = Depends(get_db)):
    user_repo = UserRepository(db)
    profile_repo = ProfileRepository(db)
    user = db.query(__import__('backend.db.models', fromlist=['User']).User).filter_by(username=username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    profile = profile_repo.get(user.id)
    if not profile:
        raise HTTPException(status_code=404, detail="No profile yet — upload a PGN first.")
    return {
        "username": username,
        "avg_cp_loss": profile.avg_cp_loss,
        "capture_rate": profile.capture_rate,
        "games_analysed": profile.games_analysed,
        "style_embedding": profile.style_embedding,
    }


@router.get("/similarity")
def style_similarity(username_a: str, username_b: str, db: Session = Depends(get_db)):
    """Cosine similarity between two users' style embeddings."""
    from backend.db.models import User
    profile_repo = ProfileRepository(db)

    def _get_profile(uname):
        u = db.query(User).filter_by(username=uname).first()
        if not u:
            raise HTTPException(status_code=404, detail=f"User '{uname}' not found.")
        p = profile_repo.get(u.id)
        if not p or not p.style_embedding:
            raise HTTPException(status_code=404, detail=f"No style profile for '{uname}'.")
        return p

    pa = _get_profile(username_a)
    pb = _get_profile(username_b)
    sim = inference_engine.cosine_similarity(pa.style_embedding, pb.style_embedding)
    return {"username_a": username_a, "username_b": username_b, "cosine_similarity": round(sim, 4)}
