"""
Gymnasium environment for NUMS game with action masking and reward shaping.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from simulator import NumsGame, SLOT_COUNT, SLOT_MAX, POWER_COUNT


NUM_ACTIONS = 23  # 18 slots + 2 select + 3 apply
OBS_SIZE = 122    # 74 base + 9 derived + 18 lookahead + 18 range + 3 phase


class NumsEnv(gym.Env):
    """NUMS game as a Gymnasium environment with action masking.

    Action space (Discrete 23):
        0-17:  place number in slot i
        18-19: select power at index 0 or 1
        20-22: apply power at index 0, 1, or 2

    Observation includes visible active trap information from the live game.
    Reward: +1 per level gained + shaped bonus (annealed over training).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None, shaping_weight=0.3):
        super().__init__()
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32
        )
        self.game = NumsGame()
        self.render_mode = render_mode
        self.shaping_weight = shaping_weight  # 0 = no shaping, 1 = full shaping
        self._prev_quality = 0.0

    def set_shaping_weight(self, weight: float):
        """Update reward shaping weight (for annealing during training)."""
        self.shaping_weight = max(0.0, min(1.0, weight))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        game_seed = self.np_random.integers(0, 2**31) if seed is None else seed
        self.game.reset(int(game_seed))
        self._prev_quality = self.game.board_quality()
        obs = np.array(self.game.get_observation(), dtype=np.float32)
        return obs, {"action_mask": np.array(self.game.action_mask(), dtype=np.bool_)}

    def step(self, action: int):
        prev_level = self.game.level
        prev_quality = self._prev_quality
        action = int(action)

        try:
            if 0 <= action <= 17:
                self.game.place(action)
            elif 18 <= action <= 19:
                self.game.select_power(action - 18)
            elif 20 <= action <= 22:
                self.game.apply_power(action - 20)
        except Exception:
            # Invalid action
            self.game.over = True
            obs = np.array(self.game.get_observation(), dtype=np.float32)
            return obs, -1.0, True, False, {
                "action_mask": np.array([False] * NUM_ACTIONS, dtype=np.bool_),
                "level": self.game.level,
            }

        # Base reward: +1 per level gained
        level_reward = float(self.game.level - prev_level)

        # Shaped reward: change in board quality
        new_quality = self.game.board_quality()
        quality_delta = new_quality - prev_quality
        self._prev_quality = new_quality

        # Bonus for keeping next_number placeable after a placement
        # Proportional signal: the model gets gradient to maximize flexibility,
        # not just avoid the worst case.
        next_bonus = 0.0
        if level_reward > 0 and not self.game.over:
            next_valid = len(self.game._valid_slots_for(self.game.next_number))
            # 0 valid → -0.20, 1 → -0.15, 2 → -0.10, 3 → -0.05,
            # 4 → 0.00, 5 → +0.05, 6+ → +0.10
            next_bonus = -0.2 + min(next_valid, 6) * 0.05

        # Penalty for needing to use a power (got stuck)
        power_penalty = 0.0
        if 20 <= action <= 22:
            power_penalty = -0.05  # small cost for burning a power

        # Combine: base + annealed shaping
        shaped = quality_delta * 0.3 + next_bonus + power_penalty
        reward = level_reward + self.shaping_weight * shaped

        done = self.game.over

        # ── End-of-game bonuses/penalties (from krump analysis) ──
        if done:
            # Unused power penalty: krump uses 100% of powers, never wastes one
            unused_powers = len(self.game.enabled_powers)
            reward -= self.shaping_weight * 0.3 * unused_powers

            # Win bonus: completing all 18 slots is a major achievement
            if self.game.is_completed():
                reward += 3.0

            # Near-win bonus: reaching 15+ is elite (krump hits 15+ in 27% of games)
            elif self.game.level >= 15:
                reward += 0.5
        obs = np.array(self.game.get_observation(), dtype=np.float32)
        mask = np.array(self.game.action_mask(), dtype=np.bool_)

        if not done and not mask.any():
            self.game.over = True
            done = True

        info = {
            "action_mask": mask,
            "level": self.game.level,
        }

        if self.render_mode == "human" and done:
            print(f"Game over at level {self.game.level}/18 | Slots: {self.game.slots}")

        return obs, reward, done, False, info

    def action_masks(self) -> np.ndarray:
        return np.array(self.game.action_mask(), dtype=np.bool_)

    def valid_action_mask(self) -> np.ndarray:
        return self.action_masks()
