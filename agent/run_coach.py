import argparse
import chess
import chess.pgn

from coach_agent import CoachAgent


def load_pgn_games(path):
    games = []
    with open(path) as f:
        while True:
            g = chess.pgn.read_game(f)
            if g is None:
                break
            games.append(g)
    return games


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pgn", required=True, help="path to a .pgn file with one or more games")
    p.add_argument("--player_name", required=True, help="name as it appears in the PGN White/Black headers")
    p.add_argument("--style_ckpt", default=None, help="optional path to project-1's chess_style_net.pt")
    p.add_argument("--max_iterations", type=int, default=8)
    args = p.parse_args()

    games = load_pgn_games(args.pgn)
    if not games:
        print("No games found in PGN file.")
        return

    # figure out which color the player was in the first game (assumes consistent across file)
    player_color = chess.WHITE if games[0].headers.get("White") == args.player_name else chess.BLACK

    agent = CoachAgent(style_ckpt_path=args.style_ckpt, max_iterations=args.max_iterations)
    report, puzzles, findings, memory = agent.run_session(args.player_name, games, player_color=player_color)

    print("\n" + "=" * 60)
    print(report)
    print("=" * 60)

    print(f"\nGenerated {len(puzzles)} practice puzzle(s):")
    for i, pz in enumerate(puzzles, 1):
        print(f"\nPuzzle {i} ({pz['phase']}): {pz['prompt']}")
        print(f"  FEN: {pz['fen']}")
        print(f"  Your move was {pz['player_move']} (lost ~{pz['cp_loss_of_player_move']}cp)")
        print(f"  Better move: {pz['best_move']}")

    style_findings = [f for f in findings if f["action"] == "style_deviation"]
    if style_findings:
        obs = style_findings[0]["observation"]
        print(f"\nStyle-model check (from project 1's ChessStyleNet) on the worst blunder:")
        print(f"  probability a 'style A' player would've played that move: {obs['style_a']:.4f}")
        print(f"  probability a 'style B' player would've played that move: {obs['style_b']:.4f}")

    print(f"\nMemory now has {len(memory['sessions'])} session(s) recorded for {args.player_name}.")
    print(f"Cumulative phase blunder counts: {memory['cumulative_phase_blunder_counts']}")


if __name__ == "__main__":
    main()
