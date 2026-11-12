"""
Persistent, cross-session memory for the Chess Coach Agent.

Deliberately simple (JSON on disk, one file per player) rather than a vector
DB -- the point of this project is the agent's planning/tool-use loop, not
infra. Swap in a real vector store later if you want that on your resume too
(see README).
"""

import json
import os
import time

MEMORY_DIR = os.path.join(os.path.dirname(__file__), "memory")


def _path(player_name: str) -> str:
    safe = "".join(c for c in player_name if c.isalnum() or c in "-_")
    return os.path.join(MEMORY_DIR, f"{safe}.json")


def load_memory(player_name: str) -> dict:
    os.makedirs(MEMORY_DIR, exist_ok=True)
    p = _path(player_name)
    if not os.path.exists(p):
        return {
            "player_name": player_name,
            "sessions": [],
            "cumulative_phase_blunder_counts": {"opening": 0, "middlegame": 0, "endgame": 0},
            "cumulative_cp_loss_by_phase": {"opening": 0, "middlegame": 0, "endgame": 0},
        }
    with open(p) as f:
        return json.load(f)


def save_session(player_name: str, findings: list, blunders: list, report: str):
    mem = load_memory(player_name)

    phase_counts = {"opening": 0, "middlegame": 0, "endgame": 0}
    phase_cp = {"opening": 0, "middlegame": 0, "endgame": 0}
    for b in blunders:
        phase_counts[b["phase"]] += 1
        phase_cp[b["phase"]] += b["cp_loss"]

    for k in phase_counts:
        mem["cumulative_phase_blunder_counts"][k] += phase_counts[k]
        mem["cumulative_cp_loss_by_phase"][k] += phase_cp[k]

    mem["sessions"].append(
        {
            "timestamp": time.time(),
            "num_blunders_found": len(blunders),
            "phase_counts_this_session": phase_counts,
            "report_summary": report[:500],
        }
    )

    with open(_path(player_name), "w") as f:
        json.dump(mem, f, indent=2)

    return mem


def weakest_phase(mem: dict) -> str:
    """Which phase has accumulated the most blunders across all sessions so far."""
    counts = mem["cumulative_phase_blunder_counts"]
    if sum(counts.values()) == 0:
        return None
    return max(counts, key=counts.get)
