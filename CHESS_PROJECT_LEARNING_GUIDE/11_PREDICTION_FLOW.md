# 11 — Complete Chess Prediction & Inference Flow

This document details the exact runtime path from the instant a user drags a chess piece in their web browser to the moment the engine responds with its style-conditioned move suggestion.

---

## 1. Step-by-Step Execution Trace

```mermaid
sequenceDiagram
    autonumber
    actor Player as User (Browser UI)
    participant UI as play.html (chessboard.js)
    participant WS as /ws/play (backend/api/routes/ws.py)
    participant Inf as InferenceEngine (backend/ml/inference.py)
    participant Redis as Redis Cache
    participant Core as board_encoding.py
    participant Net as ChessStyleNet (src/ml/model.py)

    Player->>UI: Plays Move e2-e4
    UI->>UI: chess.js updates FEN: "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    UI->>WS: Send JSON: {"type": "evaluate", "fen": "<FEN>", "top_k": 5}
    
    WS->>Inf: get_top_moves(fen, current_history, top_k=5)
    Inf->>Inf: Compute SHA256 key: "cse:a4f89d..."
    Inf->>Redis: GET "cse:a4f89d..."
    
    alt Cache Hit
        Redis-->>Inf: Return cached JSON
    else Cache Miss
        Inf->>Core: board_to_tensor(board) -> (14, 8, 8) NumPy
        Inf->>Core: legal_move_mask(board) -> (4672,) Bool Mask
        Inf->>Net: Forward(board_tensor, history_tensor)
        Net-->>Inf: Raw policy_logits (1, 4672) & blunder_logit (1, 1)
        
        Inf->>Inf: Apply Legal Move Mask (illegal logits = -1e9)
        Inf->>Inf: Softmax over legal moves -> Probabilities
        Inf->>Inf: Extract Top-5 Moves (indices -> UCI: ["e7e5", "c7c5", ...])
        Inf->>Redis: SETEX "cse:a4f89d..." 3600 (Cache result)
    end
    
    Inf-->>WS: Return {"top_moves": [...], "blunder_risk": 0.04}
    WS-->>UI: WebSocket Frame: {"type": "evaluation", "top_moves": [...], "blunder_risk": 0.04}
    UI->>UI: Render probability bars & Blunder Gauge on panel
```

---

## 2. Deep Dive: Legal Move Masking

A critical flaw in standard language/vision models applied to games is that they can predict **illegal moves** (e.g., jumping through pieces, moving into check). 

`InferenceEngine` solves this mathematically via **Logit Masking**:
1. It queries `python-chess` for all legal moves in the position:
   ```python
   legal_moves = list(board.legal_moves)
   ```
2. It builds a boolean mask vector $\mathbf{M} \in \{0, 1\}^{4672}$:
   $$M_i = \begin{cases} 1 & \text{if move } i \text{ is legal} \\ 0 & \text{otherwise} \end{cases}$$
3. For every illegal move ($M_i = 0$), the raw logit $z_i$ is replaced with a massive negative scalar:
   $$\tilde{z}_i = \begin{cases} z_i & \text{if } M_i = 1 \\ -10^9 & \text{if } M_i = 0 \end{cases}$$
4. When Softmax is computed:
   $$P(\text{move}_i) = \frac{e^{\tilde{z}_i}}{\sum_{j} e^{\tilde{z}_j}}$$
   Since $e^{-10^9} \approx 0$, the probability of selecting an illegal move becomes **identically 0.0**, while preserving the relative probability ranking among all legal choices!

