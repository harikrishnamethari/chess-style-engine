"""
The agent's "brain" -- decides what to do next given current evidence, and
writes the final natural-language report. This is the ONE swappable
component: plug in Claude or GPT via API key, or run the built-in adaptive
heuristic planner with zero setup and zero API cost.

Both brains implement the same interface:
  - decide_next_action(memory, findings, plan_state, num_games) -> dict
        {"action": str, "args": dict, "rationale": str}
  - synthesize_report(player_name, memory, findings) -> str

Why the heuristic brain still counts as "agentic":
  It is NOT a fixed script. The next action depends on what was OBSERVED so
  far -- e.g. if the middlegame is showing 2x the blunder rate of other
  phases, it decides to gather more middlegame evidence across additional
  games instead of moving on. That conditional, evidence-driven control
  flow is the actual definition of agentic behavior; the LLM call is one way
  to implement the decision function, not the only way.
"""

import json
import os

AVAILABLE_ACTIONS = [
    "analyze_phase",     # args: {"game_index": int, "phase": "opening"|"middlegame"|"endgame"}
    "next_game",         # args: {} -- move to analyzing a fresh game
    "finalize",          # args: {} -- stop gathering evidence, write the report
]


class HeuristicBrain:
    """Zero-dependency adaptive planner. Deterministic but state-dependent."""

    name = "heuristic (no API key set)"

    def decide_next_action(self, memory, findings, plan_state, num_games):
        checked = plan_state["phases_checked"]  # list of (game_index, phase) already done
        game_idx = plan_state["current_game_index"]

        # 1. Prior-session memory bias: if a phase is historically weak, check it first each session
        weak_phase = plan_state.get("weakest_phase_from_memory")
        phase_order = ["opening", "middlegame", "endgame"]
        if weak_phase and weak_phase in phase_order:
            phase_order = [weak_phase] + [p for p in phase_order if p != weak_phase]

        # 2. If current game still has unchecked phases, check the next one
        for phase in phase_order:
            if (game_idx, phase) not in checked:
                return {
                    "action": "analyze_phase",
                    "args": {"game_index": game_idx, "phase": phase},
                    "rationale": (
                        f"Prior sessions flagged '{weak_phase}' as the weakest phase, prioritizing it."
                        if phase == weak_phase and weak_phase
                        else f"Checking {phase} of game {game_idx} (not yet analyzed)."
                    ),
                }

        # 3. This game's phases are all checked. Adaptive branch: did one phase
        #    show a much higher blunder rate THIS session? If so, and games
        #    remain, look at that same phase in one more game before finalizing.
        this_game_blunders = [f for f in findings if f["observation"].get("game_index") == game_idx]
        phase_counts = {}
        for f in this_game_blunders:
            for b in f["observation"].get("blunders", []):
                phase_counts[b["phase"]] = phase_counts.get(b["phase"], 0) + 1

        if phase_counts:
            hot_phase = max(phase_counts, key=phase_counts.get)
            if phase_counts[hot_phase] >= 2 and game_idx + 1 < num_games and not plan_state.get("did_deep_dive"):
                plan_state["did_deep_dive"] = True
                return {
                    "action": "next_game",
                    "args": {},
                    "rationale": (
                        f"Game {game_idx} showed {phase_counts[hot_phase]} blunders in the {hot_phase} phase "
                        f"alone -- pulling in another game to confirm this is a pattern, not a one-off."
                    ),
                }

        # 4. Move to the next game if any remain and we haven't hit a reasonable cap
        if game_idx + 1 < num_games and plan_state["games_fully_checked"] < 2:
            return {"action": "next_game", "args": {}, "rationale": "Current game fully analyzed; moving to next game."}

        return {"action": "finalize", "args": {}, "rationale": "Sufficient evidence gathered across phases/games."}

    def synthesize_report(self, player_name, memory, findings):
        all_blunders = []
        for f in findings:
            all_blunders.extend(f["observation"].get("blunders", []))

        phase_counts = {"opening": 0, "middlegame": 0, "endgame": 0}
        for b in all_blunders:
            phase_counts[b["phase"]] += 1

        lines = [f"# Coaching Report for {player_name}", ""]
        if not all_blunders:
            lines.append("No significant blunders (>=150cp loss) detected in the analyzed games. Solid session!")
            return "\n".join(lines)

        worst_phase = max(phase_counts, key=phase_counts.get)
        lines.append(
            f"Across the analyzed games, {len(all_blunders)} blunders were found. "
            f"The **{worst_phase}** phase was the most error-prone "
            f"({phase_counts[worst_phase]} of {len(all_blunders)} blunders)."
        )
        lines.append("")
        lines.append("Breakdown by phase:")
        for phase, count in phase_counts.items():
            lines.append(f"  - {phase}: {count} blunder(s)")

        lines.append("")
        lines.append("Worst individual blunders:")
        for b in sorted(all_blunders, key=lambda x: -x["cp_loss"])[:3]:
            lines.append(
                f"  - Move {b['move_number']} ({b['phase']}): played {b['played_move']}, "
                f"lost ~{b['cp_loss']} centipawns."
            )

        if memory["sessions"]:
            lines.append("")
            lines.append(
                f"This is session #{len(memory['sessions']) + 1} for this player. "
                f"Cumulative weakest phase across all sessions so far: "
                f"**{max(memory['cumulative_phase_blunder_counts'], key=memory['cumulative_phase_blunder_counts'].get)}**."
            )

        lines.append("")
        lines.append(f"Recommendation: focus practice puzzles on the {worst_phase} phase (see generated puzzles).")
        return "\n".join(lines)


class LLMBrain:
    """Real LLM-driven planner. Requires ANTHROPIC_API_KEY or OPENAI_API_KEY."""

    def __init__(self, provider: str, api_key: str):
        self.provider = provider
        self.api_key = api_key
        self.name = f"LLM ({provider})"

    def _chat(self, system: str, user: str) -> str:
        if self.provider == "anthropic":
            import anthropic

            client = anthropic.Anthropic(api_key=self.api_key)
            resp = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1000,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return "".join(b.text for b in resp.content if hasattr(b, "text"))
        elif self.provider == "openai":
            import openai

            client = openai.OpenAI(api_key=self.api_key)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
            return resp.choices[0].message.content
        raise ValueError(f"Unknown provider {self.provider}")

    def decide_next_action(self, memory, findings, plan_state, num_games):
        system = (
            "You are the planning module of a chess coaching agent. Given the current "
            "evidence gathered so far, decide the single next action. Respond ONLY with "
            f"JSON: {{\"action\": one of {AVAILABLE_ACTIONS}, \"args\": {{...}}, \"rationale\": \"...\"}}. "
            "No other text."
        )
        user = json.dumps(
            {
                "memory_summary": memory.get("cumulative_phase_blunder_counts"),
                "findings_so_far": findings,
                "plan_state": plan_state,
                "num_games_available": num_games,
                "available_actions": AVAILABLE_ACTIONS,
            },
            default=str,
        )
        raw = self._chat(system, user).strip().strip("```json").strip("```")
        return json.loads(raw)

    def synthesize_report(self, player_name, memory, findings):
        system = (
            "You are a supportive but honest chess coach. Write a short, specific, "
            "encouraging coaching report in markdown based on the evidence provided. "
            "Reference concrete move numbers and centipawn losses where available."
        )
        user = json.dumps({"player_name": player_name, "memory": memory, "findings": findings}, default=str)
        return self._chat(system, user)


def get_brain():
    """Auto-selects the brain based on available API keys."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return LLMBrain("anthropic", os.environ["ANTHROPIC_API_KEY"])
    if os.environ.get("OPENAI_API_KEY"):
        return LLMBrain("openai", os.environ["OPENAI_API_KEY"])
    return HeuristicBrain()
