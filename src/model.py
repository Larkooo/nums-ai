"""
Policy and Value networks for NUMS using Apple MLX.

Architecture: shared feature extractor → policy head + value head
Supports action masking for variable valid actions per state.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from env import OBS_SIZE, NUM_ACTIONS

# Use large negative instead of -inf to avoid NaN in logsumexp/entropy
MASK_VALUE = -1e9


class NumsPolicy(nn.Module):
    """Actor-Critic network for NUMS with action masking."""

    def __init__(self, obs_size: int = OBS_SIZE, n_actions: int = NUM_ACTIONS, hidden: int = 256):
        super().__init__()

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(obs_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        # Policy head (actor)
        self.policy = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, n_actions),
        )

        # Value head (critic)
        self.value = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def __call__(self, obs, action_mask=None):
        features = self.shared(obs)
        logits = self.policy(features)
        value = self.value(features)

        if action_mask is not None:
            logits = mx.where(action_mask, logits, mx.array(MASK_VALUE))

        return logits, value


def compute_log_probs(logits: mx.array, actions: mx.array) -> mx.array:
    """Compute log probabilities of taken actions."""
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    # Clamp to avoid extreme values
    log_probs = mx.clip(log_probs, -20.0, 0.0)
    batch_size = actions.shape[0]
    indices = mx.arange(batch_size)
    return log_probs[indices, actions.astype(mx.int32)]


def compute_entropy(logits: mx.array) -> mx.array:
    """Compute entropy of the policy distribution (numerically stable)."""
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    probs = mx.exp(log_probs)
    # Only count valid actions (not masked out)
    valid = logits > MASK_VALUE + 1
    safe_log = mx.where(valid, log_probs, mx.zeros_like(log_probs))
    safe_prob = mx.where(valid, probs, mx.zeros_like(probs))
    entropy = -mx.sum(safe_prob * safe_log, axis=-1)
    return entropy
