# 15 — Chess Domain Concepts for Software & ML Engineers

To excel in interviews, you must demonstrate a firm grasp of both the chess domain mechanics and how they translate into machine learning representations.

---

## 1. Core Chess Terminology

### PGN (Portable Game Notation)
A standardized, human-readable plain text format for recording entire chess games. Includes metadata tags (`[White "Magnus Carlsen"]`, `[Result "1-0"]`) followed by move tokens in algebraic notation (`1. e4 e5 2. Nf3 Nc6...`).
- *In this project*: `backend/ml/pgn_parser.py` parses PGN streams to compute sequential player metrics.

### FEN (Forsyth-Edwards Notation)
A single-line string encoding an instantaneous snapshot of a board position, including:
1. Piece placements rank-by-rank (`rnbqkbnr/pppppppp/...`).
2. Active turn (`w` or `b`).
3. Castling availability (`KQkq` or `-`).
4. En passant target square (`e3` or `-`).
5. Halfmove clock (50-move rule).
6. Fullmove number.
- *In this project*: FEN strings are the communication payload across WebSockets and are converted to 14-plane tensors via `board_to_tensor()`.

### Centipawn (cp) & Centipawn Loss (CPL)
The atomic unit of chess position evaluation. $100 \text{ centipawns} = 1 \text{ pawn}$ advantage.
- If an engine evaluates a position as $+1.50$, White has an advantage roughly equivalent to 1.5 pawns.
- **Centipawn Loss**: The difference in evaluation between the best possible move in a position and the move actually played by the human:
  $$\text{CPL} = \text{Eval}(\text{Best Move}) - \text{Eval}(\text{Played Move})$$
- *In this project*: Lower average CPL indicates higher grandmaster-level positional accuracy.

---

## 2. Four Key Objectives in Chess AI: Which Does This Project Implement?

In interviews, you must clearly distinguish between four distinct chess engineering goals:

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Finding the Objectively Best Move                        │
│    (Stockfish, AlphaBeta search, MCTS evaluation)           │
│    -> NOT this project's primary goal.                      │
├─────────────────────────────────────────────────────────────┤
│ 2. Static Position Evaluation                               │
│    (Predicting win/draw/loss probability or centipawn score)│
│    -> Handled as an auxiliary blunder-detection signal.     │
├─────────────────────────────────────────────────────────────┤
│ 3. Predicting Human Moves                                   │
│    (Behavioral cloning / supervised policy training)        │
│    -> YES: Core policy head predicts move distributions.    │
├─────────────────────────────────────────────────────────────┤
│ 4. Modeling & Conditioning on Individual Player Style       │
│    (Adapting move predictions based on past user habits)    │
│    -> YES: Unique innovation of ChessStyleNet's GRU style   │
│       encoder + style interpolation engine.                 │
└─────────────────────────────────────────────────────────────┘
```

### Why this is a superior engineering project for interviews:
Building another Stockfish wrapper is trivial. Predicting *which* human move a specific player will choose—capturing their tactical greed, risk tolerance, or endgame patience—is a substantially more sophisticated behavioral modeling challenge with direct applications to recommendation systems, personalization engines, and conversational AI.

