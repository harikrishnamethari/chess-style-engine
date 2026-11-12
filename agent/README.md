# Chess Coach Agent — an agentic AI layer on top of the Style-Conditioned Chess Engine

This is project 2, built to directly extend project 1 rather than sit next to
it. It's an autonomous agent that analyzes a player's games, plans its own
investigation, remembers findings across sessions, and generates a
personalized coaching report + practice puzzles.

## How it links to project 1

- The `style_deviation` tool (`tools.py`) loads project 1's `ChessStyleNet`
  checkpoint and scores how probable a human's actual move was under
  different learned style embeddings — a model-based complement to raw
  Stockfish centipawn loss for describing *how* a player deviates from a
  given style, not just *how much* they lost.
- Both projects share the same board/move encoding (`board_encoding.py`,
  reused via `sys.path`).
- The pitch for interviews: "I built the ML core first (a style-conditioned
  policy network), then wrapped it in an autonomous agent that uses it as
  one of several tools, alongside a real chess engine, to run multi-step
  personalized coaching sessions with persistent memory."

## Architecture (Thought → Action → Observation loop)

```
agent/
├── tools.py            # stockfish_eval, find_blunders_in_phase, style_deviation, generate_puzzle
├── memory_store.py      # JSON-based persistent memory, one file per player, across sessions
├── llm_client.py         # pluggable "brain": HeuristicBrain (no API key) or LLMBrain (Claude/GPT)
├── coach_agent.py         # the orchestration loop tying brain + tools + memory together
├── run_coach.py            # CLI entrypoint
└── generate_sample_games.py # synthetic test player with an injected endgame weakness
```

Each iteration: the **brain** decides the next action given everything
observed so far (`decide_next_action`) → the loop **executes** that action
via a tool → the **observation** (blunders found, etc.) is appended to
`findings` → repeat until the brain decides to `finalize`, or a hard
iteration cap is hit (always include a safety cap in an agent loop).

### Why this counts as genuinely agentic (not a scripted pipeline)

The heuristic fallback brain's next action depends on what was *observed*:
if a game shows 2+ blunders concentrated in one phase, the agent decides on
its own to pull in another game to confirm the pattern before finalizing,
rather than following a fixed step order. Prior-session memory also
re-orders which phase gets checked first in later sessions. That
evidence-driven branching is the actual definition of agentic behavior — an
LLM call is one way to implement the decision function, not the only way.
Swap in `LLMBrain` (real Claude/GPT call, see `llm_client.py`) for
free-form planning once you have an API key; the rest of the system is
identical either way.

## What's been validated (in this sandbox, real Stockfish, no API key)

- Installed and confirmed Stockfish 16 via UCI (`/usr/games/stockfish`) —
  blunder detection uses a **real engine**, not a heuristic eval.
- Generated synthetic PGN games for a test player with a deliberately
  injected weakness (blunders far more often in the endgame than the
  opening). Ran a full coaching session:
  - agent adaptively checked phases, noticed a phase with 2+ blunders, and
    decided **on its own** to pull in a second game before finalizing
  - produced a report correctly identifying the weakest phase, the 3 worst
    blunders with exact centipawn losses, and 3 auto-generated practice
    puzzles (each with Stockfish's actual best move as the solution)
- Ran a **second session** for the same player and confirmed cross-session
  memory works: session 2 explicitly reasoned "Prior sessions flagged
  'middlegame' as the weakest phase, prioritizing it" — proving persistence
  actually changes future planning, not just logging.
- Confirmed the `style_deviation` tool correctly loads a project-1
  checkpoint and returns move probabilities under different style
  embeddings, with no crash when no checkpoint is provided (graceful
  degradation — the agent still works with Stockfish-only evidence).

## How to run it

### With the built-in heuristic planner (works immediately, no API key)
```bash
cd agent
python generate_sample_games.py sample_games.pgn
python run_coach.py --pgn sample_games.pgn --player_name test_player --style_ckpt ../src/chess_style_net.pt
```

### With a real LLM doing the planning
```bash
export ANTHROPIC_API_KEY=sk-...     # or OPENAI_API_KEY
python run_coach.py --pgn sample_games.pgn --player_name test_player
```
`llm_client.get_brain()` auto-detects the key and switches to `LLMBrain` —
no other code changes needed.

### On your own real games
Export your Lichess/chess.com games as PGN (Lichess: profile → export games)
and point `--pgn` at that file, with `--player_name` matching your username
as it appears in the PGN headers. Run it weekly and memory will accumulate a
real picture of your recurring weaknesses over time.

## Extending further

- **Vector-store memory**: swap `memory_store.py`'s JSON file for a real
  embedding-based store (e.g. embed each blunder's tactical pattern) so the
  agent can retrieve "games similar to this recurring mistake" instead of
  just phase counts — a stronger RAG-flavored story for interviews.
- **More tools**: an opening-book lookup tool, a "time trouble" detector
  using PGN clock annotations, or a tool that fetches a player's Lichess
  rating history to correlate blunder rate with time control.
- **Multi-agent**: split into a dedicated "Analyst" agent (gathers evidence)
  and a "Coach" agent (turns evidence into a lesson plan), communicating
  through the shared memory store — a natural extension if you want to
  demonstrate multi-agent coordination specifically.

## Interview pitch (one-liner)

"I built an autonomous coaching agent that plans its own multi-step
investigation into a player's games, using a real chess engine and a
custom-trained style-conditioning model as tools, with persistent
cross-session memory — and the planning module is swappable between a
zero-cost heuristic planner and a real LLM."
