"""
PPO training loop for NUMS using Apple MLX.

Usage:
    python train.py                    # Train with defaults
    python train.py --steps 2000000    # Train for 2M steps
    python train.py --eval-only        # Just evaluate saved model
"""

import argparse
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from env import NumsEnv, NUM_ACTIONS, OBS_SIZE
from model import NumsPolicy, sample_action, compute_log_probs, compute_entropy
from simulator import play_baseline_game


# ──────────────────────────────────────────────────────────────
# ANSI helpers
# ──────────────────────────────────────────────────────────────

BOLD    = "\033[1m"
DIM     = "\033[2m"
RESET   = "\033[0m"
RED     = "\033[31m"
GREEN   = "\033[32m"
YELLOW  = "\033[33m"
BLUE    = "\033[34m"
CYAN    = "\033[36m"
MAGENTA = "\033[35m"
UP      = "\033[A"
CLEAR   = "\033[2K"


def _term_width():
    try:
        return os.get_terminal_size().columns
    except Exception:
        return 80


# ──────────────────────────────────────────────────────────────
# PPO Hyperparameters
# ──────────────────────────────────────────────────────────────

DEFAULTS = dict(
    total_steps=1_000_000,
    rollout_steps=2048,      # steps per rollout before update
    n_epochs=4,              # PPO epochs per update
    batch_size=256,          # minibatch size
    gamma=0.99,              # discount factor
    gae_lambda=0.95,         # GAE lambda
    clip_eps=0.2,            # PPO clip epsilon
    vf_coef=0.5,             # value loss coefficient
    ent_coef=0.01,           # entropy bonus coefficient
    lr=3e-4,                 # learning rate
    max_grad_norm=0.5,       # gradient clipping
    hidden_size=256,         # network hidden size
    n_envs=8,                # parallel environments
    eval_interval=50_000,    # evaluate every N steps
    eval_games=1000,         # games per evaluation
    save_dir="checkpoints",
)


# ──────────────────────────────────────────────────────────────
# Live dashboard
# ──────────────────────────────────────────────────────────────

class Dashboard:
    """Live-updating terminal dashboard for training progress."""

    TOTAL_LINES = 5  # fixed dashboard height

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.t_start = time.time()
        self.last_render = 0
        self.rendered_once = False

        # Tracking
        self.total_steps = 0
        self.episodes = 0
        self.best_avg = 0.0
        self.best_eval = 0.0
        self.baseline_avg = None

        # Rolling windows
        self.recent_levels = deque(maxlen=100)
        self.recent_losses = deque(maxlen=20)
        self.recent_pg_losses = deque(maxlen=20)
        self.recent_vf_losses = deque(maxlen=20)
        self.recent_entropy = deque(maxlen=20)
        self.level_history = []   # (steps, avg_level) for sparkline
        self.sps_samples = deque(maxlen=10)

        # Current update stats
        self.cur_loss = 0.0
        self.cur_pg_loss = 0.0
        self.cur_vf_loss = 0.0
        self.cur_entropy = 0.0
        self.cur_sps = 0
        self.phase = "rollout"  # rollout | update | eval
        self.eval_nn = None
        self.eval_bl = None

    def _elapsed(self):
        return time.time() - self.t_start

    def _eta(self):
        elapsed = self._elapsed()
        if self.total_steps == 0:
            return "..."
        rate = self.total_steps / elapsed
        remaining = self.cfg["total_steps"] - self.total_steps
        secs = remaining / rate if rate > 0 else 0
        return _fmt_duration(secs)

    def _avg_level(self):
        if not self.recent_levels:
            return 0.0
        return sum(self.recent_levels) / len(self.recent_levels)

    def _sparkline(self, values, width=20):
        if len(values) < 2:
            return DIM + "..." + RESET
        # Take last `width` values
        vals = list(values)[-width:]
        mn, mx = min(vals), max(vals)
        rng = mx - mn if mx != mn else 1.0
        blocks = " ▁▂▃▄▅▆▇█"
        line = ""
        for v in vals:
            idx = int((v - mn) / rng * (len(blocks) - 1))
            line += blocks[idx]
        return line

    def _progress_bar(self, width=30):
        pct = min(self.total_steps / self.cfg["total_steps"], 1.0)
        filled = int(pct * width)
        bar = "█" * filled + "░" * (width - filled)
        return bar, pct

    def render(self, force=False):
        """Render the dashboard. Throttled to ~4 fps unless forced."""
        now = time.time()
        if not force and (now - self.last_render) < 0.25:
            return
        self.last_render = now

        # Move cursor up to overwrite previous render
        if self.rendered_once:
            sys.stdout.write(f"\033[{self.TOTAL_LINES}A")
        self.rendered_once = True

        lines = []
        bar, pct = self._progress_bar(40)
        elapsed_str = _fmt_duration(self._elapsed())
        eta_str = self._eta()
        avg_lv = self._avg_level()
        lv_color = GREEN if avg_lv >= 10 else (YELLOW if avg_lv >= 7 else RED)
        loss_str = f"{self.cur_loss:.4f}" if self.cur_loss else "  ..."
        pg_str = f"{self.cur_pg_loss:.4f}" if self.cur_pg_loss else "..."
        vf_str = f"{self.cur_vf_loss:.4f}" if self.cur_vf_loss else "..."
        ent_str = f"{self.cur_entropy:.4f}" if self.cur_entropy else "..."
        spark = self._sparkline([v for _, v in self.level_history], 30)
        best_str = f"{self.best_avg:.2f}" if self.best_avg > 0 else "..."
        eval_best = f"{GREEN}{self.best_eval:.2f}{RESET}" if self.best_eval > 0 else f"{DIM}...{RESET}"

        lines.append(f"{bar} {BOLD}{pct*100:5.1f}%{RESET} {DIM}{elapsed_str} elapsed, ETA {RESET}{BOLD}{eta_str}{RESET} {DIM}({self.phase}){RESET}")
        lines.append(f"{DIM}{self.total_steps:,}/{self.cfg['total_steps']:,} steps{RESET} {DIM}│{RESET} {self.cur_sps:,} sps {DIM}│{RESET} {self.episodes:,} episodes")
        lines.append(f"level {lv_color}{BOLD}{avg_lv:.2f}{RESET} {DIM}(best {best_str}){RESET} {DIM}│{RESET} loss {CYAN}{loss_str}{RESET} {DIM}[pg {pg_str} vf {vf_str} ent {ent_str}]{RESET}")
        lines.append(f"trend {spark}")

        if self.eval_nn is not None and self.eval_bl is not None:
            delta = self.eval_nn - self.eval_bl
            dc = GREEN if delta > 0 else (RED if delta < 0 else YELLOW)
            lines.append(f"eval  NN={CYAN}{self.eval_nn:.2f}{RESET} baseline={DIM}{self.eval_bl:.2f}{RESET} {dc}{BOLD}{delta:+.2f}{RESET} {DIM}│{RESET} best eval {eval_best}")
        else:
            lines.append(f"eval  {DIM}pending...{RESET} {DIM}│{RESET} best eval {eval_best}")

        # Pad to fixed height
        while len(lines) < self.TOTAL_LINES:
            lines.append("")

        output = "\n".join(CLEAR + line for line in lines[:self.TOTAL_LINES])
        sys.stdout.write(output + "\n")
        sys.stdout.flush()

    def print_header(self):
        """Print the static header once at start."""
        cfg = self.cfg
        print(f"{BOLD}nums neural network training{RESET} {DIM}— Proximal Policy Optimization on Apple MLX | envs={cfg['n_envs']} batch={cfg['batch_size']} lr={cfg['lr']} hidden={cfg['hidden_size']}{RESET}")
        for _ in range(self.TOTAL_LINES):
            sys.stdout.write("\n")


def _fmt_duration(secs):
    """Format seconds as human-readable duration."""
    if secs < 0:
        return "..."
    secs = int(secs)
    if secs < 60:
        return f"{secs}s"
    elif secs < 3600:
        return f"{secs // 60}m {secs % 60}s"
    else:
        h = secs // 3600
        m = (secs % 3600) // 60
        return f"{h}h {m}m"


# ──────────────────────────────────────────────────────────────
# Rollout buffer
# ──────────────────────────────────────────────────────────────

class RolloutBuffer:
    """Stores rollout data for PPO training."""

    def __init__(self, n_steps: int, n_envs: int):
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.obs = np.zeros((n_steps, n_envs, OBS_SIZE), dtype=np.float32)
        self.actions = np.zeros((n_steps, n_envs), dtype=np.int32)
        self.log_probs = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.rewards = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.dones = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.values = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.masks = np.zeros((n_steps, n_envs, NUM_ACTIONS), dtype=np.bool_)
        self.ptr = 0

    def add(self, obs, actions, log_probs, rewards, dones, values, masks):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = actions
        self.log_probs[self.ptr] = log_probs
        self.rewards[self.ptr] = rewards
        self.dones[self.ptr] = dones
        self.values[self.ptr] = values
        self.masks[self.ptr] = masks
        self.ptr += 1

    def compute_gae(self, last_values: np.ndarray, gamma: float, lam: float):
        """Compute GAE advantages and returns."""
        advantages = np.zeros_like(self.rewards)
        last_gae = 0

        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_values = last_values
            else:
                next_values = self.values[t + 1]

            next_non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_values * next_non_terminal - self.values[t]
            advantages[t] = last_gae = delta + gamma * lam * next_non_terminal * last_gae

        returns = advantages + self.values
        return advantages, returns

    def get_batches(self, advantages, returns, batch_size):
        """Yield random minibatches from the buffer."""
        total = self.n_steps * self.n_envs
        indices = np.random.permutation(total)

        # Flatten everything
        flat_obs = self.obs.reshape(total, OBS_SIZE)
        flat_actions = self.actions.reshape(total)
        flat_log_probs = self.log_probs.reshape(total)
        flat_advantages = advantages.reshape(total)
        flat_returns = returns.reshape(total)
        flat_masks = self.masks.reshape(total, NUM_ACTIONS)

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            idx = indices[start:end]
            yield (
                mx.array(flat_obs[idx]),
                mx.array(flat_actions[idx]),
                mx.array(flat_log_probs[idx]),
                mx.array(flat_advantages[idx]),
                mx.array(flat_returns[idx]),
                mx.array(flat_masks[idx]),
            )


# ──────────────────────────────────────────────────────────────
# Vectorized environment wrapper
# ──────────────────────────────────────────────────────────────

class VecEnv:
    """Simple vectorized environment (serial execution)."""

    def __init__(self, n_envs: int):
        self.envs = [NumsEnv() for _ in range(n_envs)]
        self.n_envs = n_envs

    def reset(self):
        obs_list = []
        mask_list = []
        for env in self.envs:
            obs, info = env.reset()
            obs_list.append(obs)
            mask_list.append(info["action_mask"])
        return np.array(obs_list), np.array(mask_list)

    def step(self, actions):
        obs_list, reward_list, done_list, mask_list, level_list = [], [], [], [], []

        for i, (env, action) in enumerate(zip(self.envs, actions)):
            obs, reward, done, truncated, info = env.step(action)
            level_list.append(info["level"])
            if done:
                final_level = info["level"]
                obs, new_info = env.reset()
                info["action_mask"] = new_info["action_mask"]
                info["final_level"] = final_level
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            mask_list.append(info["action_mask"])

        return (
            np.array(obs_list),
            np.array(reward_list, dtype=np.float32),
            np.array(done_list, dtype=np.float32),
            np.array(mask_list),
            level_list,
        )


# ──────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────

def train(cfg: dict):
    save_dir = Path(cfg["save_dir"])
    save_dir.mkdir(exist_ok=True)

    # Create model and optimizer
    model = NumsPolicy(hidden=cfg["hidden_size"])
    mx.eval(model.parameters())
    optimizer = optim.Adam(learning_rate=cfg["lr"])

    # Create environments
    vec_env = VecEnv(cfg["n_envs"])
    obs, masks = vec_env.reset()

    # Dashboard
    dash = Dashboard(cfg)
    dash.print_header()

    # Tracking
    log_data = []
    t_start = time.time()
    steps_per_rollout = cfg["rollout_steps"] * cfg["n_envs"]

    rollout_num = 0

    while rollout_num * steps_per_rollout < cfg["total_steps"]:
        # ── Collect rollout ──
        dash.phase = "rollout"
        buffer = RolloutBuffer(cfg["rollout_steps"], cfg["n_envs"])
        rollout_base = rollout_num * steps_per_rollout

        for step in range(cfg["rollout_steps"]):
            obs_mx = mx.array(obs)
            masks_mx = mx.array(masks)

            logits, values = model(obs_mx, masks_mx)
            mx.eval(logits, values)

            # Sample actions for each env
            actions = np.zeros(cfg["n_envs"], dtype=np.int32)
            log_probs_np = np.zeros(cfg["n_envs"], dtype=np.float32)

            logits_np = np.array(logits)
            values_np = np.array(values).squeeze(-1)

            for i in range(cfg["n_envs"]):
                action_idx, log_prob = _sample_action_np(logits_np[i])
                actions[i] = action_idx
                log_probs_np[i] = log_prob

            # Step environments
            next_obs, rewards, dones, next_masks, levels = vec_env.step(actions)

            # Track completed episodes
            for i in range(cfg["n_envs"]):
                if dones[i]:
                    dash.episodes += 1
                    dash.recent_levels.append(levels[i])

            buffer.add(obs, actions, log_probs_np, rewards, dones, values_np, masks)
            obs = next_obs
            masks = next_masks

            # Update dashboard during rollout (every 128 steps)
            if step % 128 == 0:
                dash.total_steps = rollout_base + step * cfg["n_envs"]
                elapsed = time.time() - t_start
                dash.cur_sps = int(dash.total_steps / elapsed) if elapsed > 0 else 0
                dash.render()

        rollout_num += 1
        dash.total_steps = rollout_num * steps_per_rollout

        # ── Compute advantages ──
        dash.phase = "update"
        dash.render(force=True)

        _, last_values = model(mx.array(obs), mx.array(masks))
        mx.eval(last_values)
        last_values_np = np.array(last_values).squeeze(-1)

        advantages, returns = buffer.compute_gae(last_values_np, cfg["gamma"], cfg["gae_lambda"])

        # Normalize advantages
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        # ── PPO update ──
        total_loss = 0.0
        total_pg = 0.0
        total_vf = 0.0
        total_ent = 0.0
        n_updates = 0

        loss_and_grad_fn = nn.value_and_grad(model, _ppo_loss)

        for epoch in range(cfg["n_epochs"]):
            for batch in buffer.get_batches(advantages, returns, cfg["batch_size"]):
                b_obs, b_actions, b_old_log_probs, b_advantages, b_returns, b_masks = batch

                (loss, (pg_l, vf_l, ent_l)), grads = loss_and_grad_fn(
                    model, b_obs, b_actions, b_old_log_probs,
                    b_advantages, b_returns, b_masks,
                    cfg["clip_eps"], cfg["vf_coef"], cfg["ent_coef"],
                )
                mx.eval(loss, grads)

                # Gradient clipping
                grads = _clip_grad_norm(grads, cfg["max_grad_norm"])

                optimizer.update(model, grads)
                mx.eval(model.parameters(), optimizer.state)

                n_updates += 1
                total_loss += loss.item()
                total_pg += pg_l.item()
                total_vf += vf_l.item()
                total_ent += ent_l.item()

                # Live loss updates during PPO
                dash.cur_loss = total_loss / n_updates
                dash.cur_pg_loss = total_pg / n_updates
                dash.cur_vf_loss = total_vf / n_updates
                dash.cur_entropy = total_ent / n_updates
                dash.phase = f"update {epoch+1}/{cfg['n_epochs']}"
                dash.render()

        # Update dashboard metrics
        if n_updates > 0:
            dash.cur_loss = total_loss / n_updates
            dash.cur_pg_loss = total_pg / n_updates
            dash.cur_vf_loss = total_vf / n_updates
            dash.cur_entropy = total_ent / n_updates
            dash.recent_losses.append(dash.cur_loss)
            dash.recent_pg_losses.append(dash.cur_pg_loss)
            dash.recent_vf_losses.append(dash.cur_vf_loss)
            dash.recent_entropy.append(dash.cur_entropy)

        elapsed = time.time() - t_start
        dash.cur_sps = int(dash.total_steps / elapsed) if elapsed > 0 else 0

        avg_lv = dash._avg_level()
        if avg_lv > dash.best_avg:
            dash.best_avg = avg_lv
        dash.level_history.append((dash.total_steps, avg_lv))

        # Log data
        log_data.append({
            "steps": dash.total_steps,
            "avg_level_100": round(avg_lv, 2),
            "episodes": dash.episodes,
            "loss": round(dash.cur_loss, 4),
            "pg_loss": round(dash.cur_pg_loss, 4),
            "vf_loss": round(dash.cur_vf_loss, 4),
            "entropy": round(dash.cur_entropy, 4),
            "sps": dash.cur_sps,
        })

        dash.render(force=True)

        # ── Periodic evaluation ──
        if dash.total_steps % cfg["eval_interval"] < steps_per_rollout:
            dash.phase = "eval"
            dash.render(force=True)

            eval_avg = evaluate_model(model, cfg["eval_games"])
            baseline_avg = evaluate_baseline(cfg["eval_games"])

            dash.eval_nn = eval_avg
            dash.eval_bl = baseline_avg

            if eval_avg > dash.best_eval:
                dash.best_eval = eval_avg
                save_model(model, save_dir / "best_model.npz")

            dash.render(force=True)

    # Save final model
    save_model(model, save_dir / "final_model.npz")
    with open(save_dir / "training_log.json", "w") as f:
        json.dump(log_data, f, indent=2)

    # Final summary (below the dashboard)
    dash.phase = "done"
    dash.render(force=True)
    print()
    print(f"  {GREEN}{BOLD}Training complete!{RESET}")
    print(f"  {dash.episodes:,} episodes in {_fmt_duration(time.time() - t_start)}")
    print(f"  Best eval avg: {dash.best_eval:.2f}")
    print(f"  Model saved to {save_dir}/")
    print()

    return model


# ──────────────────────────────────────────────────────────────
# PPO Loss (returns sub-losses for dashboard)
# ──────────────────────────────────────────────────────────────

def _ppo_loss(
    model, obs, actions, old_log_probs, advantages, returns, masks,
    clip_eps, vf_coef, ent_coef,
):
    """Compute PPO clipped surrogate loss + value loss + entropy bonus."""
    logits, values = model(obs, masks)
    values = values.squeeze(-1)

    # Policy loss
    new_log_probs = compute_log_probs(logits, actions)
    ratio = mx.exp(new_log_probs - old_log_probs)
    clipped_ratio = mx.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    pg_loss = -mx.minimum(ratio * advantages, clipped_ratio * advantages).mean()

    # Value loss
    vf_loss = mx.mean((values - returns) ** 2)

    # Entropy bonus
    entropy = compute_entropy(logits).mean()

    total = pg_loss + vf_coef * vf_loss - ent_coef * entropy
    return total, (pg_loss, vf_loss, entropy)


# ──────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────

def _sample_action_np(logits: np.ndarray):
    """Sample action from numpy logits (handles -inf masking)."""
    valid = logits > -1e30
    if not valid.any():
        return 0, 0.0

    max_logit = logits[valid].max()
    shifted = np.where(valid, logits - max_logit, -1e30)
    exp_shifted = np.where(valid, np.exp(shifted), 0.0)
    sum_exp = exp_shifted.sum()
    probs = exp_shifted / sum_exp

    action = np.random.choice(len(probs), p=probs)
    log_prob = np.log(probs[action] + 1e-10)
    return int(action), float(log_prob)


def _clip_grad_norm(grads, max_norm):
    """Clip gradient norm (tree of arrays)."""
    flat, treedef = mx.utils.tree_flatten(grads)
    total_norm_sq = sum(mx.sum(g * g).item() for g in flat if g is not None)
    total_norm = total_norm_sq ** 0.5
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-6)
        flat = [g * scale if g is not None else g for g in flat]
    return mx.utils.tree_unflatten(list(zip([k for k, _ in mx.utils.tree_flatten(grads)], flat)))


def evaluate_model(model, n_games: int = 1000) -> float:
    """Play n_games using the NN model and return average level."""
    levels = []
    env = NumsEnv()

    for i in range(n_games):
        obs, info = env.reset(seed=i + 100000)
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
            action = int(np.argmax(logits_np))

            obs, reward, done, truncated, info = env.step(action)

        levels.append(env.game.level)

    return sum(levels) / len(levels)


def evaluate_baseline(n_games: int = 1000) -> float:
    """Play n_games using baseline bot and return average level."""
    levels = [play_baseline_game(seed=i + 100000) for i in range(n_games)]
    return sum(levels) / len(levels)


def save_model(model, path: Path):
    """Save model weights."""
    model.save_weights(str(path))


def load_model(path: Path, hidden: int = 256):
    """Load model weights."""
    model = NumsPolicy(hidden=hidden)
    model.load_weights(str(path))
    return model


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train NUMS NN with PPO (MLX)")
    parser.add_argument("--steps", type=int, default=DEFAULTS["total_steps"])
    parser.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    parser.add_argument("--hidden", type=int, default=DEFAULTS["hidden_size"])
    parser.add_argument("--n-envs", type=int, default=DEFAULTS["n_envs"])
    parser.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--rollout-steps", type=int, default=DEFAULTS["rollout_steps"])
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-games", type=int, default=DEFAULTS["eval_games"])
    parser.add_argument("--save-dir", type=str, default=DEFAULTS["save_dir"])
    args = parser.parse_args()

    if args.eval_only:
        path = Path(args.save_dir) / "best_model.npz"
        if not path.exists():
            path = Path(args.save_dir) / "final_model.npz"
        if not path.exists():
            print(f"No model found in {args.save_dir}/")
            return

        model = load_model(path, hidden=args.hidden)
        nn_avg = evaluate_model(model, args.eval_games)
        bl_avg = evaluate_baseline(args.eval_games)
        print(f"NN avg level:       {nn_avg:.2f}")
        print(f"Baseline avg level: {bl_avg:.2f}")
        print(f"Improvement:        {nn_avg - bl_avg:+.2f}")
        return

    cfg = dict(DEFAULTS)
    cfg.update(
        total_steps=args.steps,
        lr=args.lr,
        hidden_size=args.hidden,
        n_envs=args.n_envs,
        batch_size=args.batch_size,
        rollout_steps=args.rollout_steps,
        eval_games=args.eval_games,
        save_dir=args.save_dir,
    )

    train(cfg)


if __name__ == "__main__":
    main()
