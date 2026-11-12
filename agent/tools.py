"""
Tools available to the Chess Coach Agent. Each tool is a plain function with
a docstring and a typed signature -- this is deliberately framework-free so
you can see exactly what "tool use" means under the hood before reaching for
LangGraph/CrewAI abstractions in a resume-polished version.

Tools:
  - stockfish_eval_cp(fen)              : centipawn evaluation of a position
  - find_blunders_in_phase(...)         : real-engine blunder detection for a
                                           phase of a game (opening/middlegame/endgame)
  - style_deviation(...)                : reuses ChessStyleNet from project 1 --
                                           how "un-expert-like" was this move?
  - generate_puzzle(...)                : turns a detected blunder into a
                                           "find the better move" puzzle
"""

import os
import sys
import chess
import chess.engine
import chess.pgn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

STOCKFISH_PATH = os.environ.get("STOCKFISH_PATH", "/usr/games/stockfish")

PHASE_RANGES = {
    "opening": (1, 10),
    "middlegame": (11, 30),
    "endgame": (31, 10_000),
}

BLUNDER_THRESHOLD_CP = 150  # centipawn loss above this = blunder, matches common Lichess convention


class Engine:
    """Thin context-managed wrapper so callers don't have to manage the UCI process."""

    def __init__(self, path=STOCKFISH_PATH, depth=12):
        self.path = path
        self.depth = depth
        self._engine = None

    def __enter__(self):
        self._engine = chess.engine.SimpleEngine.popen_uci(self.path)
        return self

    def __exit__(self, *args):
        if self._engine:
            self._engine.quit()

    def eval_cp(self, board: chess.Board, mover_pov=True) -> int:
        """Centipawn evaluation of `board`, from the perspective of the side to move
        (positive = good for the side to move). Mate scores are clipped to +/-1000."""
        info = self._engine.analyse(board, chess.engine.Limit(depth=self.depth))
        score = info["score"].pov(board.turn if mover_pov else chess.WHITE)
        return score.score(mate_score=1000)

    def best_move(self, board: chess.Board):
        result = self._engine.play(board, chess.engine.Limit(depth=self.depth))
        return result.move


def find_blunders_in_phase(engine: Engine, game: chess.pgn.Game, phase: str, player_color: chess.Color):
    """
    Walk through `game`'s moves that (a) fall in the given phase's move-number
    range and (b) were played by `player_color`, evaluating before/after with
    Stockfish. Returns a list of blunder dicts.
    """
    lo, hi = PHASE_RANGES[phase]
    board = game.board()
    blunders = []
    move_number = 0

    for move in game.mainline_moves():
        move_number += 1  # counts plies; fine-grained enough for phase bucketing
        is_players_move = board.turn == player_color
        in_phase = lo <= move_number <= hi

        if is_players_move and in_phase:
            eval_before = engine.eval_cp(board)  # from mover's POV, before playing
            board.push(move)
            eval_after_from_opponent_pov = engine.eval_cp(board)  # now opponent to move
            eval_after_from_mover_pov = -eval_after_from_opponent_pov
            cp_loss = eval_before - eval_after_from_mover_pov

            if cp_loss >= BLUNDER_THRESHOLD_CP:
                blunders.append(
                    {
                        "phase": phase,
                        "move_number": move_number,
                        "fen_before": board.fen(),  # NOTE: fen AFTER push, used as puzzle start below via undo
                        "played_move": move.uci(),
                        "cp_loss": cp_loss,
                    }
                )
                # store the pre-move fen properly
                board.pop()
                blunders[-1]["fen_before"] = board.fen()
                board.push(move)
        else:
            board.push(move)

    return blunders


def style_deviation(fen: str, played_move_uci: str, ckpt_path: str, style_id_a: int, style_id_b: int):
    """
    Reuses project 1's ChessStyleNet: returns the probability the model
    (under two different learned styles) assigns to the move the human
    actually played. A low probability under an "expert/strong" style id
    means the move deviates a lot from strong play -- a personalized,
    model-based alternative/complement to raw centipawn loss.

    Returns None gracefully if no checkpoint is available yet (e.g. before
    project 1's Colab training has been run) so the agent can still operate
    with Stockfish-only evidence.
    """
    if not os.path.exists(ckpt_path):
        return None

    import torch
    from board_encoding import board_to_tensor, move_to_index
    from model import ChessStyleNet

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = ChessStyleNet(num_players=ckpt["num_players"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    board = chess.Board(fen)
    move = chess.Move.from_uci(played_move_uci)
    move_idx = move_to_index(move, board)
    board_t = torch.from_numpy(board_to_tensor(board)).unsqueeze(0)

    result = {}
    with torch.no_grad():
        for label, pid in [("style_a", style_id_a), ("style_b", style_id_b)]:
            pid_t = torch.tensor([pid])
            logits, _ = model(board_t, player_ids=pid_t)
            probs = torch.softmax(logits.squeeze(0), dim=0)
            result[label] = probs[move_idx].item()
    return result


def generate_puzzle(engine: Engine, blunder: dict):
    """Turn a detected blunder into a 'find the better move' puzzle using
    Stockfish's actual best move as the solution."""
    board = chess.Board(blunder["fen_before"])
    best = engine.best_move(board)
    return {
        "fen": blunder["fen_before"],
        "phase": blunder["phase"],
        "prompt": f"It's {'White' if board.turn else 'Black'} to move. Find a better move than {blunder['played_move']}.",
        "player_move": blunder["played_move"],
        "best_move": best.uci(),
        "cp_loss_of_player_move": blunder["cp_loss"],
    }
