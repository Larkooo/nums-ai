"""
Gymnasium environment for NUMS game with action masking support.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from simulator import NumsGame, SLOT_COUNT, SLOT_MAX, POWER_COUNT


NUM_ACTIONS = 23  # 18 slots + 2 select + 3 apply
OBS_SIZE = 29


class NumsEnv(gym.Env):
    """NUMS game as a Gymnasium environment with action masking.

    Action space (Discrete 23):
        0-17:  place number in slot i
        18-19: select power at index 0 or 1
        20-22: apply power at index 0, 1, or 2

    Observation space (Box 29):
        [0:18]  - 18 slots normalized to [0, 1]
        [18]    - current number / 999
        [19]    - next number / 999
        [20]    - level / 18
        [21:23] - selectable power types (normalized)
        [23:29] - selected power types + enabled flags (3 × 2)

    Reward: +1 per level gained (dense signal).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        super().__init__()
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32
        )
        self.game = NumsGame()
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        game_seed = self.np_random.integers(0, 2**31) if seed is None else seed
        self.game.reset(int(game_seed))
        obs = np.array(self.game.get_observation(), dtype=np.float32)
        return obs, {"action_mask": np.array(self.game.action_mask(), dtype=np.bool_)}

    def step(self, action: int):
        prev_level = self.game.level
        action = int(action)

        try:
            if 0 <= action <= 17:
                self.game.place(action)
            elif 18 <= action <= 19:
                self.game.select_power(action - 18)
            elif 20 <= action <= 22:
                self.game.apply_power(action - 20)
        except (AssertionError, Exception):
            # Invalid action → game over with penalty
            self.game.over = True
            obs = np.array(self.game.get_observation(), dtype=np.float32)
            return obs, -1.0, True, False, {
                "action_mask": np.array([False] * NUM_ACTIONS, dtype=np.bool_),
                "level": self.game.level,
            }

        # Reward = levels gained this step (0 or 1)
        reward = float(self.game.level - prev_level)
        done = self.game.over
        obs = np.array(self.game.get_observation(), dtype=np.float32)
        mask = np.array(self.game.action_mask(), dtype=np.bool_)

        # If no valid actions, game is over
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
        """For sb3-contrib MaskablePPO compatibility."""
        return np.array(self.game.action_mask(), dtype=np.bool_)

    def valid_action_mask(self) -> np.ndarray:
        """Alias for action_masks()."""
        return self.action_masks()


def make_env(seed=None, render_mode=None):
    """Create a NumsEnv instance."""
    def _init():
        env = NumsEnv(render_mode=render_mode)
        return env
    return _init
