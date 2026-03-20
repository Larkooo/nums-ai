# NUMS AI

Neural network that learns to play [NUMS](https://nums.gg) — a fully on-chain number-placement strategy game on Starknet.

Trained with Proximal Policy Optimization (PPO) on Apple MLX. Reaches **10.13 avg level** vs 9.05 baseline (+1.08 improvement).

## Results

| Agent | Avg Level | vs Baseline |
|-------|-----------|-------------|
| Baseline (proportional mapping) | 9.05 | — |
| **NUMS AI (PPO, 5M steps)** | **10.13** | **+1.08** |

Pre-trained models are available in `models/`.

## The Game

NUMS is a push-your-luck game: place randomly drawn numbers (1-999) into 18 ascending-order slots. Features:
- **5 hidden traps** (Bomb, Lucky, Magnet, UFO, Windy) that fire on placement and chain-react
- **7 powers** (Reroll, High, Low, Swap, DoubleUp, Halve, Mirror) drawn at levels 4, 8, 12
- **Win condition**: fill all 18 slots in ascending order

## Project Structure

```
src/
├── simulator.py    # Faithful port of the Cairo on-chain game logic
├── env.py          # Gymnasium environment with action masking
├── model.py        # Actor-Critic MLP on Apple MLX
├── train.py        # PPO training with live dashboard
├── evaluate.py     # NN vs baseline comparison
└── demo.py         # TUI demo — watch the AI play
models/
└── model_eval_10.13.npz   # Best pre-trained model
```

## Quick Start

Requires Python 3.10+ and Apple Silicon (for MLX).

```bash
pip install mlx numpy gymnasium
```

### Train

```bash
cd src
python train.py --steps 5000000
```

### Resume from checkpoint

```bash
python train.py --steps 10000000 --resume ../models/model_eval_10.13.npz
```

### Evaluate

```bash
python evaluate.py --games 5000
```

### Watch the AI play

```bash
# With trained model
python demo.py --model ../models/model_eval_10.13.npz --speed 1.0

# Compare NN vs baseline on the same game
python demo.py --model ../models/model_eval_10.13.npz --side-by-side

# Baseline only (no MLX needed)
python demo.py
```

## Architecture

**Observation** (83 features): 18 slot values, current/next number, level, 18 occupancy flags, one-hot power encodings, plus 9 derived features (valid slot counts for current and next number, min gap, board fill ratio, ideal position, stuck flag).

**Actions** (23 discrete, masked): 18 slot placements + 2 power selections + 3 power applications. Invalid actions are masked out so the NN only chooses from legal moves.

**Training**: PPO with GAE, action masking, gradient clipping. 16 parallel environments, 512 batch size, 6 PPO epochs per rollout. Live terminal dashboard shows progress, loss breakdown, entropy, and periodic eval vs baseline.

## How It Learns

1. Plays thousands of games in parallel, exploring random strategies
2. Gets +1 reward per level reached
3. PPO computes advantages ("was this action better or worse than expected?") using a critic network
4. Updates policy to favor actions that led to higher-than-expected outcomes
5. Entropy bonus keeps it exploring instead of committing too early
6. Gradually discovers strategies like: good spacing, next-number lookahead, strategic power usage
