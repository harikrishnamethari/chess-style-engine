"""
CoachAgent: the orchestration loop. Thought -> Action -> Observation, repeated
until the brain decides to finalize (or a hard iteration cap is hit as a
safety net against infinite loops -- always include this in an agent).
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import chess
from tools import Engine, find_blunders_in_phase, generate_puzzle, style_deviation
from memory_store import load_memory, save_session, weakest_phase
from llm_client import get_brain


class CoachAgent:
    def __init__(self, style_ckpt_path: str = None, max_iterations: int = 8):
        self.brain = get_brain()
        self.style_ckpt_path = style_ckpt_path
        self.max_iterations = max_iterations
        print(f"[CoachAgent] planning brain: {self.brain.name}")

    def run_session(self, player_name: str, games: list, player_color=chess.WHITE):
        """
        games: list of chess.pgn.Game objects (this player's games to analyze)
        Returns: (report_str, puzzles_list, findings_list, updated_memory)
        """
        memory = load_memory(player_name)
        plan_state = {
            "phases_checked": [],
            "current_game_index": 0,
            "games_fully_checked": 0,
            "weakest_phase_from_memory": weakest_phase(memory),
            "did_deep_dive": False,
        }
        findings = []
        all_blunders = []

        with Engine() as engine:
            for step in range(self.max_iterations):
                decision = self.brain.decide_next_action(memory, findings, plan_state, len(games))
                print(f"  step {step+1}: [{decision['action']}] {decision['rationale']}")

                if decision["action"] == "finalize":
                    break

                elif decision["action"] == "analyze_phase":
                    g_idx = decision["args"]["game_index"]
                    phase = decision["args"]["phase"]
                    blunders = find_blunders_in_phase(engine, games[g_idx], phase, player_color)
                    plan_state["phases_checked"].append((g_idx, phase))
                    all_blunders.extend(blunders)
                    findings.append(
                        {
                            "thought": decision["rationale"],
                            "action": decision["action"],
                            "observation": {"game_index": g_idx, "phase": phase, "blunders": blunders},
                        }
                    )
                    if all(
                        (g_idx, p) in plan_state["phases_checked"]
                        for p in ["opening", "middlegame", "endgame"]
                    ):
                        plan_state["games_fully_checked"] += 1

                elif decision["action"] == "next_game":
                    plan_state["current_game_index"] += 1
                    findings.append(
                        {"thought": decision["rationale"], "action": decision["action"], "observation": {}}
                    )

            else:
                print(f"  [CoachAgent] hit max_iterations={self.max_iterations} safety cap, finalizing anyway.")

            report = self.brain.synthesize_report(player_name, memory, findings)

            puzzles = []
            for b in sorted(all_blunders, key=lambda x: -x["cp_loss"])[:3]:
                puzzles.append(generate_puzzle(engine, b))

        # optional: style-deviation scoring for the worst blunder, if a trained checkpoint exists
        if self.style_ckpt_path and all_blunders:
            worst = max(all_blunders, key=lambda x: x["cp_loss"])
            dev = style_deviation(worst["fen_before"], worst["played_move"], self.style_ckpt_path, 0, 1)
            if dev:
                findings.append({"thought": "Scoring worst blunder against style model.", "action": "style_deviation", "observation": dev})

        updated_memory = save_session(player_name, findings, all_blunders, report)
        return report, puzzles, findings, updated_memory
