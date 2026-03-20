"""
Evaluate and compare NN agent vs baseline bot on NUMS.

Usage:
    python evaluate.py                          # Compare best model vs baseline
    python evaluate.py --games 10000            # More games for statistical significance
    python evaluate.py --model checkpoints/final_model.npz
"""

import argparse
import time
from collections import Counter
from pathlib import Path

import numpy as np

from simulator import play_baseline_game, NumsGame
from env import NumsEnv
from train import load_model, NumsPolicy

import mlx.core as mx


def evaluate_agent(model: NumsPolicy, n_games: int, greedy: bool = True) -> dict:
    """Evaluate NN agent. Returns stats dict."""
    levels = []
    env = NumsEnv()

    t0 = time.time()
    for i in range(n_games):
        obs, info = env.reset(seed=i)
        done = False
        while not done:
            obs_mx = mx.array(obs[np.newaxis, :])
            mask_mx = mx.array(info["action_mask"][np.newaxis, :])
            logits, _ = model(obs_mx, mask_mx)
            mx.eval(logits)

            logits_np = np.array(logits[0])
            valid = logits_np > -1e30
            if not valid.any():
                break

            if greedy:
                action = int(np.argmax(logits_np))
            else:
                # Stochastic: sample from distribution
                shifted = np.where(valid, logits_np - logits_np[valid].max(), -1e30)
                probs = np.where(valid, np.exp(shifted), 0.0)
                probs /= probs.sum()
                action = int(np.random.choice(len(probs), p=probs))

            obs, reward, done, truncated, info = env.step(action)

        levels.append(env.game.level)

    elapsed = time.time() - t0
    return _compute_stats(levels, elapsed, "NN Agent")


def evaluate_baseline(n_games: int) -> dict:
    """Evaluate baseline bot. Returns stats dict."""
    t0 = time.time()
    levels = [play_baseline_game(seed=i) for i in range(n_games)]
    elapsed = time.time() - t0
    return _compute_stats(levels, elapsed, "Baseline")


def _compute_stats(levels: list[int], elapsed: float, name: str) -> dict:
    arr = np.array(levels)
    dist = Counter(levels)
    return {
        "name": name,
        "games": len(levels),
        "avg": float(arr.mean()),
        "std": float(arr.std()),
        "median": float(np.median(arr)),
        "max": int(arr.max()),
        "min": int(arr.min()),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "completed": int(sum(1 for l in levels if l >= 18)),
        "elapsed": elapsed,
        "distribution": dict(sorted(dist.items())),
    }


def print_stats(stats: dict):
    print(f"\n{'=' * 50}")
    print(f"  {stats['name']} — {stats['games']:,} games ({stats['elapsed']:.1f}s)")
    print(f"{'=' * 50}")
    print(f"  Average level:  {stats['avg']:.3f} ± {stats['std']:.3f}")
    print(f"  Median:         {stats['median']:.0f}")
    print(f"  Range:          [{stats['min']}, {stats['max']}]")
    print(f"  P25/P75/P90:    {stats['p25']:.0f} / {stats['p75']:.0f} / {stats['p90']:.0f}")
    print(f"  Completed (18): {stats['completed']} ({stats['completed']/stats['games']*100:.3f}%)")
    print(f"  Level distribution:")
    for level in sorted(stats["distribution"]):
        count = stats["distribution"][level]
        pct = count / stats["games"] * 100
        bar = "█" * int(pct * 2)
        print(f"    Lv {level:2d}: {count:5d} ({pct:5.1f}%) {bar}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate NUMS NN vs Baseline")
    parser.add_argument("--model", type=str, default="checkpoints/best_model.npz")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--games", type=int, default=5000)
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic policy instead of greedy")
    args = parser.parse_args()

    # Evaluate baseline
    print(f"Evaluating baseline over {args.games:,} games...")
    bl_stats = evaluate_baseline(args.games)
    print_stats(bl_stats)

    # Evaluate NN if model exists
    model_path = Path(args.model)
    if not model_path.exists():
        # Try fallback
        model_path = Path("checkpoints/final_model.npz")

    if model_path.exists():
        print(f"\nLoading model from {model_path}")
        model = load_model(model_path, hidden=args.hidden)

        print(f"Evaluating NN agent over {args.games:,} games...")
        nn_stats = evaluate_agent(model, args.games, greedy=not args.stochastic)
        print_stats(nn_stats)

        # Comparison
        delta = nn_stats["avg"] - bl_stats["avg"]
        pct = (delta / bl_stats["avg"]) * 100
        print(f"\n{'=' * 50}")
        print(f"  COMPARISON")
        print(f"{'=' * 50}")
        print(f"  NN avg:       {nn_stats['avg']:.3f}")
        print(f"  Baseline avg: {bl_stats['avg']:.3f}")
        print(f"  Improvement:  {delta:+.3f} ({pct:+.1f}%)")
        print(f"  NN completions:       {nn_stats['completed']}")
        print(f"  Baseline completions: {bl_stats['completed']}")
    else:
        print(f"\nNo model found at {args.model}. Train first with: python train.py")


if __name__ == "__main__":
    main()
