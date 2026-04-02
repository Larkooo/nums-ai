#!/usr/bin/env python3
"""
NUMS AI TUI Demo — watch the neural network play NUMS in real time.

Shows the board, the AI's probability distribution over actions,
its reasoning for each decision, and trap/power events as they happen.

Usage:
    python demo.py                        # Play with baseline bot
    python demo.py --model checkpoints/best_model.npz  # Play with trained NN
    python demo.py --speed 0.5            # Slower (seconds per step)
    python demo.py --games 5              # Play 5 games
    python demo.py --seed 42              # Reproducible game
    python demo.py --side-by-side         # NN vs baseline on same seed
"""

import argparse
import os
import sys
import time
from pathlib import Path

from simulator import (
    NumsGame, PowerType, TrapType,
    SLOT_COUNT, SLOT_MIN, SLOT_MAX,
    baseline_decide_slot, baseline_select_power, baseline_apply_power,
)

# Optional: MLX + numpy for NN mode
try:
    import mlx.core as mx
    import numpy as np
    from model import NumsPolicy
    HAS_MLX = True
except ImportError:
    HAS_MLX = False


# ──────────────────────────────────────────────────────────────
# Colors (ANSI)
# ──────────────────────────────────────────────────────────────

RESET   = "\033[0m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
RED     = "\033[31m"
GREEN   = "\033[32m"
YELLOW  = "\033[33m"
BLUE    = "\033[34m"
MAGENTA = "\033[35m"
CYAN    = "\033[36m"
WHITE   = "\033[37m"
BG_GREEN  = "\033[42m"
BG_YELLOW = "\033[43m"
BG_RED    = "\033[41m"
BG_BLUE   = "\033[44m"
BG_MAGENTA = "\033[45m"

POWER_COLORS = {
    PowerType.REROLL: CYAN,
    PowerType.HIGH: RED,
    PowerType.LOW: BLUE,
    PowerType.SWAP: YELLOW,
    PowerType.DOUBLE_UP: MAGENTA,
    PowerType.HALVE: GREEN,
    PowerType.MIRROR: WHITE,
}

POWER_ICONS = {
    PowerType.REROLL: "🎲",
    PowerType.HIGH: "⬆",
    PowerType.LOW: "⬇",
    PowerType.SWAP: "🔄",
    PowerType.DOUBLE_UP: "✖2",
    PowerType.HALVE: "½",
    PowerType.MIRROR: "🪞",
}

TRAP_ICONS = {
    TrapType.BOMB: "💣",
    TrapType.LUCKY: "🍀",
    TrapType.MAGNET: "🧲",
    TrapType.UFO: "🛸",
    TrapType.WINDY: "💨",
}


def power_name(p: int) -> str:
    names = {1: "Reroll", 2: "High", 3: "Low", 4: "Swap", 5: "DoubleUp", 6: "Halve", 7: "Mirror"}
    return names.get(p, "???")


def power_colored(p: int) -> str:
    color = POWER_COLORS.get(p, WHITE)
    icon = POWER_ICONS.get(p, "?")
    return f"{color}{BOLD}{icon} {power_name(p)}{RESET}"


def trap_name(t: int) -> str:
    names = {1: "Bomb", 2: "Lucky", 3: "Magnet", 4: "UFO", 5: "Windy"}
    return names.get(t, "")


def trap_short(t: int) -> str:
    names = {1: "BMB", 2: "LCK", 3: "MAG", 4: "UFO", 5: "WND"}
    return names.get(t, "---")


# ──────────────────────────────────────────────────────────────
# Board rendering
# ──────────────────────────────────────────────────────────────

def render_board(game: NumsGame, highlight_slot: int = -1, valid_slots: list[int] = None) -> list[str]:
    """Render the game board as a list of strings."""
    lines = []
    if valid_slots is None:
        valid_slots = []

    # Header
    lines.append(f"  {DIM}╔{'═' * 58}╗{RESET}")
    lines.append(f"  {DIM}║{RESET}  {BOLD}NUMS{RESET}  "
                 f"Level {BOLD}{game.level}{RESET}/{SLOT_COUNT}  "
                 f"Current: {BOLD}{YELLOW}{game.number:>3}{RESET}  "
                 f"Next: {DIM}{game.next_number:>3}{RESET}"
                 f"{'  ' * 3}{DIM}║{RESET}")
    lines.append(f"  {DIM}╠{'═' * 58}╣{RESET}")

    # Slots - 2 rows of 9
    for row in range(2):
        slot_parts = []
        trap_parts = []
        idx_parts = []
        for col in range(9):
            i = row * 9 + col
            val = game.slots[i]
            trap = game.traps[i]
            trap_active = trap != TrapType.NONE and i not in game.disabled_traps
            if i == highlight_slot:
                if val != 0:
                    slot_parts.append(f"{BG_GREEN}{BOLD}{val:>3}{RESET}")
                else:
                    slot_parts.append(f"{BG_GREEN}{BOLD}{'---':>3}{RESET}")
            elif val != 0:
                slot_parts.append(f"{GREEN}{BOLD}{val:>3}{RESET}")
            elif i in valid_slots:
                slot_parts.append(f"{YELLOW}{DIM}{'···':>3}{RESET}")
            else:
                slot_parts.append(f"{DIM}{'···':>3}{RESET}")
            if trap_active:
                trap_parts.append(f"{MAGENTA}{DIM}{trap_short(trap):>3}{RESET}")
            else:
                trap_parts.append(f"{DIM}{'---':>3}{RESET}")
            idx_parts.append(f"{DIM}{i:>3}{RESET}")

        lines.append(f"  {DIM}║{RESET} {'│'.join(slot_parts)} {DIM}║{RESET}")
        lines.append(f"  {DIM}║{RESET} {'│'.join(trap_parts)} {DIM}║{RESET}")
        lines.append(f"  {DIM}║{RESET} {'│'.join(idx_parts)} {DIM}║{RESET}")
        if row == 0:
            lines.append(f"  {DIM}║{'─' * 58}║{RESET}")

    lines.append(f"  {DIM}╠{'═' * 58}╣{RESET}")

    # Powers section
    powers_str = ""
    if game.selected_powers:
        parts = []
        for i, p in enumerate(game.selected_powers):
            enabled = "●" if i in game.enabled_powers else "○"
            parts.append(f"{enabled} {power_colored(p)}")
        powers_str = "  ".join(parts)
    else:
        powers_str = f"{DIM}none{RESET}"

    lines.append(f"  {DIM}║{RESET}  Powers: {powers_str}{'  ' * 2}{DIM}║{RESET}")
    lines.append(f"  {DIM}║{RESET}  Trap key: {MAGENTA}{DIM}BMB{RESET}/{MAGENTA}{DIM}LCK{RESET}/{MAGENTA}{DIM}MAG{RESET}/{MAGENTA}{DIM}UFO{RESET}/{MAGENTA}{DIM}WND{RESET}{' ' * 11}{DIM}║{RESET}")
    lines.append(f"  {DIM}╚{'═' * 58}╝{RESET}")

    return lines


def render_probability_bar(probs: dict[int, float], action_labels: dict[int, str], top_n: int = 6) -> list[str]:
    """Render a horizontal bar chart of action probabilities."""
    lines = []
    sorted_actions = sorted(probs.items(), key=lambda x: -x[1])[:top_n]

    lines.append(f"  {BOLD}AI Thinking:{RESET}")
    max_bar = 30
    for action, prob in sorted_actions:
        if prob < 0.001:
            continue
        bar_len = int(prob * max_bar)
        bar = "█" * bar_len + "░" * (max_bar - bar_len)
        label = action_labels.get(action, f"action {action}")
        pct = prob * 100

        # Color by confidence
        if prob > 0.5:
            color = GREEN
        elif prob > 0.2:
            color = YELLOW
        else:
            color = DIM

        lines.append(f"    {color}{bar}{RESET} {pct:5.1f}%  {label}")

    return lines


# ──────────────────────────────────────────────────────────────
# NN Agent
# ──────────────────────────────────────────────────────────────

class NNAgent:
    def __init__(self, model_path: str, hidden: int = 256):
        if not HAS_MLX:
            raise RuntimeError("MLX not installed. Install with: pip install mlx numpy")
        self.model = NumsPolicy(hidden=hidden)
        self.model.load_weights(model_path)
        mx.eval(self.model.parameters())

    def decide(self, game: NumsGame) -> tuple[int, dict[int, float]]:
        """Return (action_index, probability_dict)."""
        obs = np.array(game.get_observation(), dtype=np.float32)
        mask = np.array(game.action_mask(), dtype=np.bool_)

        obs_mx = mx.array(obs[np.newaxis, :])
        mask_mx = mx.array(mask[np.newaxis, :])

        logits, value = self.model(obs_mx, mask_mx)
        mx.eval(logits, value)

        logits_np = np.array(logits[0])
        value_np = float(np.array(value[0, 0]))

        # Compute probabilities
        valid = logits_np > -1e30
        if not valid.any():
            return 0, {}

        shifted = np.where(valid, logits_np - logits_np[valid].max(), -1e30)
        exp_shifted = np.where(valid, np.exp(shifted), 0.0)
        probs_np = exp_shifted / exp_shifted.sum()

        probs = {i: float(probs_np[i]) for i in range(len(probs_np)) if probs_np[i] > 0.001}

        # Sample from distribution
        action = int(np.random.choice(len(probs_np), p=probs_np))

        return action, probs, value_np


class BaselineAgent:
    def decide(self, game: NumsGame):
        """Return (action_index, probability_dict, value_estimate)."""
        phase = game.get_phase()
        if phase == "select":
            idx = baseline_select_power(game)
            action = 18 + idx
            return action, {action: 1.0}, 0.0
        elif phase == "place":
            slot = baseline_decide_slot(game)
            # Show proportional mapping reasoning
            ideal = round((game.number - SLOT_MIN) / (SLOT_MAX - SLOT_MIN) * (SLOT_COUNT - 1))
            valid = game.valid_slots()
            probs = {}
            for s in valid:
                probs[s] = 0.05
            probs[slot] = max(0.5, 1.0 - len(valid) * 0.05)
            # Normalize
            total = sum(probs.values())
            probs = {k: v / total for k, v in probs.items()}
            return slot, probs, 0.0
        elif phase == "apply":
            idx = baseline_apply_power(game)
            action = 20 + idx
            return action, {action: 1.0}, 0.0
        return 0, {}, 0.0


# ──────────────────────────────────────────────────────────────
# TUI Game Runner
# ──────────────────────────────────────────────────────────────

def action_label(action: int, game: NumsGame) -> str:
    """Human-readable label for an action."""
    if 0 <= action <= 17:
        return f"Place {game.number} → slot {action}"
    elif 18 <= action <= 19:
        idx = action - 18
        if idx < len(game.selectable_powers):
            p = game.selectable_powers[idx]
            return f"Select {power_name(p)}"
        return f"Select power #{idx}"
    elif 20 <= action <= 22:
        idx = action - 20
        if idx < len(game.selected_powers):
            p = game.selected_powers[idx]
            return f"Apply {power_name(p)}"
        return f"Apply power #{idx}"
    return f"??? ({action})"


def clear_screen():
    os.system("clear" if os.name != "nt" else "cls")


def run_game(agent, seed: int, speed: float = 1.0, label: str = "AI") -> int:
    """Run a single game with TUI display. Returns final level."""
    game = NumsGame()
    game.reset(seed)

    step = 0
    events = []  # recent events for display

    while not game.over:
        clear_screen()

        phase = game.get_phase()
        if phase == "over":
            break

        valid_slots = game.valid_slots() if phase == "place" else []

        # Get AI decision
        result = agent.decide(game)
        action, probs = result[0], result[1]
        value = result[2] if len(result) > 2 else 0.0

        # Snapshot state before action
        slots_before = game.slots[:]
        number_before = game.number

        # Build action labels for probability display
        labels = {}
        for a in probs:
            labels[a] = action_label(a, game)

        # ── Render ──
        print()
        print(f"  {DIM}{'─' * 58}{RESET}")
        print(f"  {BOLD}{label}{RESET}  Game seed: {seed}  Step: {step}  "
              f"Value: {CYAN}{value:.2f}{RESET}")
        print(f"  {DIM}{'─' * 58}{RESET}")
        print()

        # Board
        highlight = action if 0 <= action <= 17 else -1
        for line in render_board(game, highlight_slot=-1, valid_slots=valid_slots):
            print(line)

        print()

        # Phase indicator
        if phase == "select":
            p0 = power_colored(game.selectable_powers[0]) if len(game.selectable_powers) > 0 else "?"
            p1 = power_colored(game.selectable_powers[1]) if len(game.selectable_powers) > 1 else "?"
            print(f"  {BOLD}⚡ Power Draw!{RESET}  Choose: [{p0}]  or  [{p1}]")
            print()
        elif phase == "apply":
            enabled_list = sorted(game.enabled_powers)
            parts = [power_colored(game.selected_powers[i]) for i in enabled_list if i < len(game.selected_powers)]
            print(f"  {BOLD}{RED}✋ Stuck!{RESET}  Number {YELLOW}{BOLD}{game.number}{RESET} has no valid slot")
            print(f"  Available powers: {', '.join(parts)}")
            print()

        # Probability bars
        if probs:
            for line in render_probability_bar(probs, labels):
                print(line)
            print()

        # Decision
        chosen_label = action_label(action, game)
        print(f"  {BOLD}→ Decision:{RESET} {GREEN}{chosen_label}{RESET}")
        print()

        # Recent events
        if events:
            print(f"  {DIM}Recent:{RESET}")
            for ev in events[-3:]:
                print(f"    {DIM}{ev}{RESET}")
            print()

        time.sleep(speed * 0.6)

        # ── Execute action ──
        event = ""
        try:
            if 0 <= action <= 17:
                # Check for trap before placing
                trap_at_slot = game.traps[action] if action not in game.disabled_traps else 0
                game.place(action)

                if trap_at_slot and trap_at_slot != TrapType.NONE:
                    icon = TRAP_ICONS.get(trap_at_slot, "⚠")
                    event = f"{icon} Trap! {trap_name(trap_at_slot)} triggered at slot {action}"

                # Check what changed
                changes = []
                for i in range(SLOT_COUNT):
                    if game.slots[i] != slots_before[i] and i != action:
                        if slots_before[i] == 0:
                            changes.append(f"slot {i}: → {game.slots[i]}")
                        elif game.slots[i] == 0:
                            changes.append(f"slot {i}: {slots_before[i]} → empty")
                        else:
                            changes.append(f"slot {i}: {slots_before[i]} → {game.slots[i]}")
                if changes:
                    event += f"  ({', '.join(changes)})"

            elif 18 <= action <= 19:
                game.select_power(action - 18)
                p = game.selected_powers[-1]
                event = f"Selected {power_name(p)}"

            elif 20 <= action <= 22:
                idx = action - 20
                p = game.selected_powers[idx]
                game.apply_power(idx)
                event = f"Applied {power_name(p)}: number {number_before} → {game.number}"

        except Exception as e:
            event = f"Error: {e}"
            game.over = True

        if event:
            events.append(event)

        step += 1

        # Show result of action briefly
        clear_screen()
        print()
        print(f"  {DIM}{'─' * 58}{RESET}")
        print(f"  {BOLD}{label}{RESET}  Game seed: {seed}  Step: {step}  "
              f"Value: {CYAN}{value:.2f}{RESET}")
        print(f"  {DIM}{'─' * 58}{RESET}")
        print()

        highlight = action if 0 <= action <= 17 else -1
        for line in render_board(game, highlight_slot=highlight, valid_slots=[]):
            print(line)

        print()
        if event:
            print(f"  {CYAN}{event}{RESET}")
        print(f"  {BOLD}→{RESET} {GREEN}{chosen_label}{RESET}")
        print()

        time.sleep(speed * 0.4)

    # ── Game over screen ──
    clear_screen()
    print()
    print(f"  {DIM}{'─' * 58}{RESET}")
    print(f"  {BOLD}{label}{RESET}  Game seed: {seed}")
    print(f"  {DIM}{'─' * 58}{RESET}")
    print()

    for line in render_board(game):
        print(line)
    print()

    if game.level >= SLOT_COUNT:
        print(f"  {BG_GREEN}{BOLD} 🎉 PERFECT GAME! All {SLOT_COUNT} slots filled! {RESET}")
    else:
        filled = sum(1 for s in game.slots if s != 0)
        print(f"  {BOLD}Game Over{RESET} — Level {BOLD}{game.level}{RESET}/{SLOT_COUNT} "
              f"({filled} slots filled)")

    print()
    if events:
        print(f"  {DIM}Event log:{RESET}")
        for ev in events:
            print(f"    {DIM}• {ev}{RESET}")
        print()

    return game.level


def run_side_by_side(nn_agent, seed: int, speed: float = 1.0):
    """Run NN and baseline on the same seed, show results."""
    print(f"\n  {BOLD}Side-by-side comparison{RESET}  (seed: {seed})")
    print(f"  {DIM}{'─' * 40}{RESET}\n")

    # NN plays
    print(f"  {CYAN}{BOLD}Neural Network playing...{RESET}")
    time.sleep(0.5)
    nn_level = run_game(nn_agent, seed, speed, label="🧠 Neural Net")

    input(f"\n  {DIM}Press Enter to see baseline play the same game...{RESET}")

    # Baseline plays same seed
    baseline_agent = BaselineAgent()
    bl_level = run_game(baseline_agent, seed, speed, label="📐 Baseline")

    # Summary
    print(f"\n  {BOLD}{'═' * 40}{RESET}")
    print(f"  {BOLD}Results (seed {seed}):{RESET}")
    print(f"    🧠 Neural Net: Level {BOLD}{nn_level}{RESET}")
    print(f"    📐 Baseline:   Level {BOLD}{bl_level}{RESET}")
    delta = nn_level - bl_level
    if delta > 0:
        print(f"    {GREEN}{BOLD}NN wins by {delta} levels!{RESET}")
    elif delta < 0:
        print(f"    {RED}Baseline wins by {-delta} levels{RESET}")
    else:
        print(f"    {YELLOW}Tie!{RESET}")
    print()


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NUMS AI TUI Demo")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained model weights (.npz)")
    parser.add_argument("--hidden", type=int, default=256,
                        help="Hidden size (must match training)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Seconds per step (default: 1.0)")
    parser.add_argument("--games", type=int, default=1,
                        help="Number of games to play")
    parser.add_argument("--seed", type=int, default=None,
                        help="Game seed (random if not set)")
    parser.add_argument("--side-by-side", action="store_true",
                        help="Compare NN vs baseline on same seeds")
    args = parser.parse_args()

    # Pick agent
    if args.model:
        if not HAS_MLX:
            print("MLX not installed. Install with: pip install mlx numpy")
            print("Falling back to baseline bot.")
            agent = BaselineAgent()
            label = "📐 Baseline"
        else:
            path = Path(args.model)
            if not path.exists():
                print(f"Model not found: {path}")
                sys.exit(1)
            print(f"Loading model from {path}...")
            agent = NNAgent(str(path), hidden=args.hidden)
            label = "🧠 Neural Net"
    else:
        agent = BaselineAgent()
        label = "📐 Baseline"
        if args.side_by_side:
            print("Need --model for side-by-side mode")
            sys.exit(1)

    import random as pyrandom
    levels = []

    for i in range(args.games):
        seed = args.seed if args.seed is not None else pyrandom.randint(0, 2**31)

        if args.side_by_side and HAS_MLX and args.model:
            run_side_by_side(agent, seed, args.speed)
        else:
            level = run_game(agent, seed, args.speed, label=label)
            levels.append(level)

        if i < args.games - 1:
            try:
                input(f"  {DIM}Press Enter for next game ({i+2}/{args.games})...{RESET}")
            except (EOFError, KeyboardInterrupt):
                break

    # Summary
    if len(levels) > 1:
        avg = sum(levels) / len(levels)
        print(f"\n  {BOLD}Summary ({len(levels)} games):{RESET}")
        print(f"    Average level: {avg:.1f}")
        print(f"    Best:          {max(levels)}")
        print(f"    Worst:         {min(levels)}")
        print()


if __name__ == "__main__":
    main()
