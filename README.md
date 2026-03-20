# nums-ai

Neural network that learns to play [NUMS](https://nums.gg) — a fully on-chain number-placement strategy game on Starknet.

## Game

NUMS is a push-your-luck game where you place randomly drawn numbers (1-999) into 18 ascending-order slots. The game features hidden traps (Bomb, Lucky, Magnet, UFO, Windy) and selectable powers (Reroll, High, Low, Swap, DoubleUp, Halve, Mirror) drawn at levels 4, 8, and 12.

## Architecture

- **Simulator** (`simulator.py`): Faithful Python port of the Cairo on-chain game logic including all trap chain reactions and power mechanics
- **Environment** (`env.py`): Gymnasium-compatible environment with action masking (23 discrete actions: 18 slot placements + 2 power selections + 3 power applications)
- **Model** (`model.py`): Actor-Critic MLP with shared feature extraction, built on Apple MLX
- **Training** (`train.py`): PPO with GAE, action masking, gradient clipping
- **Evaluation** (`evaluate.py`): Side-by-side comparison with the baseline proportional-mapping bot

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.10+ and Apple Silicon (for MLX).

## Usage

```bash
# Benchmark the baseline bot (no ML dependencies needed)
python simulator.py

# Train the neural network
python train.py --steps 1000000

# Evaluate NN vs baseline
python evaluate.py --games 5000

# Evaluate a saved model
python train.py --eval-only
```

## How it works

The NN observes: 18 normalized slot values, current/next number, level, available powers, and enabled powers (29 features total). It outputs action probabilities over 23 possible actions, masked to only valid moves.

PPO training runs thousands of simulated games in parallel, giving +1 reward per successful placement. The network learns slot placement strategy, power selection timing, and power application decisions simultaneously.

## Baseline comparison

The baseline bot uses proportional mapping (place number at its proportional position across slots) with a next-number tiebreaker — matching the strategy from the [nums-bot](https://github.com/Larkooo/nums-bot) Rust implementation.
