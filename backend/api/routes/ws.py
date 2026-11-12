"""
backend/api/routes/ws.py
WebSocket endpoint for real-time engine inference streaming.
Single Responsibility: manages WebSocket lifecycle and message routing only.
"""
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from backend.ml.inference import engine as inference_engine
from backend.core.config import settings

router = APIRouter(tags=["websocket"])

# Default neutral style history — 10 games × 3 features
_DEFAULT_HISTORY = [[25.0, 0.12, 15.0]] * settings.hist_seq_len


@router.websocket("/ws/play")
async def play_websocket(websocket: WebSocket):
    """
    Real-time inference over WebSocket.
    
    Client sends JSON messages:
      { "type": "set_style",  "history": [[f1, f2, f3], ...] }
      { "type": "evaluate",   "fen": "<FEN string>", "top_k": 5 }
    
    Server sends JSON messages:
      { "type": "evaluation", "fen": "...", "top_moves": [...], "blunder_risk": 0.1 }
      { "type": "error",      "message": "..." }
    """
    await websocket.accept()
    current_history = list(_DEFAULT_HISTORY)

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"type": "error", "message": "Invalid JSON"}))
                continue

            msg_type = msg.get("type")

            if msg_type == "set_style":
                history = msg.get("history")
                if history and isinstance(history, list):
                    current_history = history
                    await websocket.send_text(json.dumps({"type": "info", "message": "Style updated"}))

            elif msg_type == "evaluate":
                fen = msg.get("fen")
                top_k = int(msg.get("top_k", 5))
                if not fen:
                    await websocket.send_text(json.dumps({"type": "error", "message": "Missing 'fen'"}))
                    continue
                try:
                    result = inference_engine.get_top_moves(fen, current_history, top_k=top_k)
                    await websocket.send_text(json.dumps({
                        "type": "evaluation",
                        "fen": fen,
                        **result,
                    }))
                except Exception as e:
                    await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))

            else:
                await websocket.send_text(json.dumps({
                    "type": "error", "message": f"Unknown message type: '{msg_type}'"
                }))

    except WebSocketDisconnect:
        pass  # client disconnected cleanly
