"""
Policy and Value networks for NUMS using Apple MLX.

Architecture: shared feature extractor → policy head + value head
Supports action masking for variable valid actions per state.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from env import OBS_SIZE, NUM_ACTIONS


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

    def __call__(self, obs: mx.array, action_mask: mx.array | None = None):
        """Forward pass.

        Args:
            obs: (batch, obs_size) observation tensor
            action_mask: (batch, n_actions) boolean mask (True = valid)

        Returns:
            logits: (batch, n_actions) masked logits for policy
            value: (batch, 1) state value estimate
        """
        features = self.shared(obs)
        logits = self.policy(features)
        value = self.value(features)

        # Apply action mask: set invalid actions to -inf
        if action_mask is not None:
            # Where mask is False (invalid), set logits to -inf
            logits = mx.where(action_mask, logits, mx.array(float("-inf")))

        return logits, value


def sample_action(logits: mx.array) -> tuple[int, float]:
    """Sample an action from logits using Gumbel-max trick.

    Returns (action_index, log_probability).
    """
    # Compute log probabilities
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

    # Gumbel-max sampling
    gumbel_noise = -mx.log(-mx.log(mx.random.uniform(shape=logits.shape) + 1e-20) + 1e-20)
    action = mx.argmax(logits + gumbel_noise, axis=-1)

    action_idx = action.item()
    log_prob = log_probs[action_idx].item()

    return action_idx, log_prob


def compute_log_probs(logits: mx.array, actions: mx.array) -> mx.array:
    """Compute log probabilities of taken actions.

    Args:
        logits: (batch, n_actions)
        actions: (batch,) integer action indices

    Returns:
        log_probs: (batch,) log probability of each action
    """
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    # Gather log probs for taken actions
    batch_size = actions.shape[0]
    indices = mx.arange(batch_size)
    return log_probs[indices, actions.astype(mx.int32)]


def compute_entropy(logits: mx.array) -> mx.array:
    """Compute entropy of the policy distribution.

    Args:
        logits: (batch, n_actions) - may contain -inf for masked actions

    Returns:
        entropy: (batch,) entropy per sample
    """
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    probs = mx.exp(log_probs)
    # Mask out -inf entries (they contribute 0 to entropy)
    valid = logits > float("-inf")
    entropy = -mx.sum(mx.where(valid, probs * log_probs, mx.zeros_like(probs)), axis=-1)
    return entropy
