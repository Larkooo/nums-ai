"""
PPO training loop for NUMS using Apple MLX.

Usage:
    python train.py                    # Train with defaults
    python train.py --steps 2000000    # Train for 2M steps
    python train.py --eval-only        # Just evaluate saved model
"""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from env import NumsEnv, NUM_ACTIONS, OBS_SIZE
from model import NumsPolicy, sample_action, compute_log_probs, compute_entropy
from simulator import play_baseline_game


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

    # Tracking
    total_steps = 0
    episode_levels = []
    best_avg = 0.0
    log_data = []

    print(f"Training NUMS NN with PPO ({cfg['total_steps']:,} steps)")
    print(f"  Envs: {cfg['n_envs']} | Rollout: {cfg['rollout_steps']} | Batch: {cfg['batch_size']}")
    print(f"  LR: {cfg['lr']} | Hidden: {cfg['hidden_size']} | Clip: {cfg['clip_eps']}")
    print()

    t_start = time.time()

    while total_steps < cfg["total_steps"]:
        # ── Collect rollout ──
        buffer = RolloutBuffer(cfg["rollout_steps"], cfg["n_envs"])

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
                    episode_levels.append(levels[i])

            buffer.add(obs, actions, log_probs_np, rewards, dones, values_np, masks)
            obs = next_obs
            masks = next_masks

        total_steps += cfg["rollout_steps"] * cfg["n_envs"]

        # ── Compute advantages ──
        with mx.no_grad():
            _, last_values = model(mx.array(obs), mx.array(masks))
            mx.eval(last_values)
        last_values_np = np.array(last_values).squeeze(-1)

        advantages, returns = buffer.compute_gae(last_values_np, cfg["gamma"], cfg["gae_lambda"])

        # Normalize advantages
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        # ── PPO update ──
        total_pg_loss = 0
        total_vf_loss = 0
        total_entropy = 0
        n_updates = 0

        loss_and_grad_fn = nn.value_and_grad(model, _ppo_loss)

        for epoch in range(cfg["n_epochs"]):
            for batch in buffer.get_batches(advantages, returns, cfg["batch_size"]):
                b_obs, b_actions, b_old_log_probs, b_advantages, b_returns, b_masks = batch

                loss, grads = loss_and_grad_fn(
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
                total_pg_loss += loss.item()

        # ── Logging ──
        avg_loss = total_pg_loss / max(n_updates, 1)
        recent_levels = episode_levels[-100:] if episode_levels else [0]
        avg_level = sum(recent_levels) / len(recent_levels)
        elapsed = time.time() - t_start
        sps = total_steps / elapsed

        entry = {
            "steps": total_steps,
            "avg_level_100": round(avg_level, 2),
            "episodes": len(episode_levels),
            "loss": round(avg_loss, 4),
            "sps": int(sps),
        }
        log_data.append(entry)
        print(
            f"Steps: {total_steps:>8,} | "
            f"Avg Level (100): {avg_level:5.2f} | "
            f"Episodes: {len(episode_levels):>6,} | "
            f"Loss: {avg_loss:7.4f} | "
            f"SPS: {sps:,.0f}"
        )

        # ── Periodic evaluation ──
        if total_steps % cfg["eval_interval"] < cfg["rollout_steps"] * cfg["n_envs"]:
            eval_avg = evaluate_model(model, cfg["eval_games"])
            baseline_avg = evaluate_baseline(cfg["eval_games"])
            improvement = eval_avg - baseline_avg
            print(f"  ► Eval ({cfg['eval_games']} games): NN={eval_avg:.2f} | Baseline={baseline_avg:.2f} | Δ={improvement:+.2f}")

            if eval_avg > best_avg:
                best_avg = eval_avg
                save_model(model, save_dir / "best_model.npz")
                print(f"  ► New best model saved ({best_avg:.2f})")

    # Save final model
    save_model(model, save_dir / "final_model.npz")
    with open(save_dir / "training_log.json", "w") as f:
        json.dump(log_data, f, indent=2)

    print(f"\nTraining complete! {len(episode_levels):,} episodes in {time.time() - t_start:.0f}s")
    print(f"Best average level: {best_avg:.2f}")
    return model


# ──────────────────────────────────────────────────────────────
# PPO Loss
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

    return pg_loss + vf_coef * vf_loss - ent_coef * entropy


# ──────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────

def _sample_action_np(logits: np.ndarray) -> tuple[int, float]:
    """Sample action from numpy logits (handles -inf masking)."""
    # Compute stable log-softmax
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


def evaluate_model(model: NumsPolicy, n_games: int = 1000) -> float:
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

            # Greedy action (argmax)
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


def save_model(model: NumsPolicy, path: Path):
    """Save model weights."""
    model.save_weights(str(path))


def load_model(path: Path, hidden: int = 256) -> NumsPolicy:
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
