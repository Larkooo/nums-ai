"""
Faithful Python port of the NUMS on-chain game logic.

Replicates the Cairo contract mechanics including:
- 18 slots, numbers 1-999, strict ascending order
- 5 traps (Bomb, Lucky, Magnet, UFO, Windy) with chain reactions
- 7 powers (Reroll, High, Low, Swap, DoubleUp, Halve, Mirror)
- Power draws at levels 4, 8, 12
- Game-over detection with power rescue consideration
"""

import random as pyrandom
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional


# Constants (matching Cairo contracts/src/constants.cairo)
SLOT_COUNT = 18
SLOT_MIN = 1
SLOT_MAX = 999
TRAP_COUNT = 5
POWER_COUNT = 7
DRAW_COUNT = 2      # 2 powers offered per draw
DRAW_STAGE = 4      # powers drawn every 4 levels
MAX_DRAW_LEVEL = 15  # no draws at or above level 15


class TrapType(IntEnum):
    NONE = 0
    BOMB = 1
    LUCKY = 2
    MAGNET = 3
    UFO = 4
    WINDY = 5


class PowerType(IntEnum):
    NONE = 0
    REROLL = 1
    HIGH = 2
    LOW = 3
    SWAP = 4
    DOUBLE_UP = 5
    HALVE = 6
    MIRROR = 7


@dataclass
class NumsGame:
    """Complete NUMS game simulator."""

    slots: list[int] = field(default_factory=lambda: [0] * SLOT_COUNT)
    number: int = 0
    next_number: int = 0
    level: int = 0
    over: bool = False

    # Traps: list of TrapType for each slot (hidden from player)
    traps: list[int] = field(default_factory=lambda: [0] * SLOT_COUNT)
    disabled_traps: set[int] = field(default_factory=set)

    # Powers
    selectable_powers: list[int] = field(default_factory=list)  # 2 power types offered
    selected_powers: list[int] = field(default_factory=list)    # up to 3 chosen powers
    enabled_powers: set[int] = field(default_factory=set)       # indices into selected_powers

    rng: pyrandom.Random = field(default_factory=pyrandom.Random)

    def reset(self, seed: Optional[int] = None) -> "NumsGame":
        """Start a new game."""
        if seed is not None:
            self.rng = pyrandom.Random(seed)
        else:
            self.rng = pyrandom.Random()

        self.slots = [0] * SLOT_COUNT
        self.level = 0
        self.over = False
        self.disabled_traps = set()
        self.selectable_powers = []
        self.selected_powers = []
        self.enabled_powers = set()

        # Draw initial numbers
        self.number = self._draw_unique([])
        self.next_number = self._draw_unique([self.number])

        # Generate traps
        self.traps = self._generate_traps()

        return self

    # ------------------------------------------------------------------
    # Public actions (what the player/bot calls)
    # ------------------------------------------------------------------

    def place(self, slot_index: int) -> bool:
        """Place current number at slot_index. Returns True if successful."""
        assert not self.over, "Game is over"
        assert not self.selectable_powers, "Must select a power first"
        assert 0 <= slot_index < SLOT_COUNT
        assert self.slots[slot_index] == 0, f"Slot {slot_index} is occupied"

        # Place the number (may trigger trap chain)
        self._place_number(self.number, slot_index)

        # Validate ascending order
        assert self._is_valid(), "Slots not in ascending order after placement"

        # Update game state
        self._update()
        return True

    def select_power(self, index: int) -> None:
        """Select one of the offered powers (index 0 or 1)."""
        assert not self.over, "Game is over"
        assert self.selectable_powers, "No powers to select"
        assert 0 <= index < len(self.selectable_powers)

        power_type = self.selectable_powers[index]
        self.selected_powers.append(power_type)
        self.selectable_powers = []

        # Enable the power (index in selected_powers)
        power_idx = len(self.selected_powers) - 1
        self.enabled_powers.add(power_idx)

        # Check game over
        self._check_game_over()

    def apply_power(self, power_index: int) -> None:
        """Apply an enabled power by its index in selected_powers."""
        assert not self.over, "Game is over"
        assert not self.selectable_powers, "Must select a power first"
        assert power_index in self.enabled_powers, f"Power {power_index} not enabled"

        # Disable the power
        self.enabled_powers.discard(power_index)
        power_type = self.selected_powers[power_index]

        # Apply the power's effect
        self._apply_power_effect(power_type)

        # Check game over
        self._check_game_over()

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def valid_slots(self) -> list[int]:
        """Return indices of slots where current number can be placed."""
        return self._valid_slots_for(self.number)

    def is_completed(self) -> bool:
        return self.level >= SLOT_COUNT

    def get_phase(self) -> str:
        """Return current game phase: 'select', 'place', 'apply', or 'over'."""
        if self.over:
            return "over"
        if self.selectable_powers:
            return "select"
        if self._valid_slots_for(self.number):
            return "place"
        if self.enabled_powers:
            return "apply"
        return "over"  # stuck with no options

    def action_mask(self) -> list[bool]:
        """Return a 23-element boolean mask for valid actions.

        Actions 0-17:  place in slot i
        Actions 18-19: select power index 0 or 1
        Actions 20-22: apply power index 0, 1, or 2
        """
        mask = [False] * 23

        if self.over:
            return mask

        if self.selectable_powers:
            for i in range(len(self.selectable_powers)):
                mask[18 + i] = True
            return mask

        valid = self._valid_slots_for(self.number)
        if valid:
            for s in valid:
                mask[s] = True
            return mask

        if self.enabled_powers:
            for idx in self.enabled_powers:
                if idx < 3:
                    mask[20 + idx] = True
            return mask

        return mask

    def get_observation(self) -> list[float]:
        """Return observation vector.

        Layout (83 features):
          [0:18]  - 18 slots normalized to [0, 1] (0 = empty)
          [18]    - current number / 999
          [19]    - next number / 999
          [20]    - level / 18
          [21:39] - 18 binary flags: is slot occupied?
          [39:46] - selectable power 0 one-hot (7 types)
          [46:53] - selectable power 1 one-hot (7 types)
          [53:60] - enabled power 0 one-hot (7 types, zeros if not enabled)
          [60:67] - enabled power 1 one-hot
          [67:74] - enabled power 2 one-hot
          --- derived features ---
          [74]    - valid slots for current number / 18
          [75]    - valid slots for next number / 18
          [76]    - min gap between adjacent filled numbers / 999 (board tightness)
          [77]    - fraction of board filled
          [78]    - number of enabled powers / 3
          [79]    - ideal slot position for current number / 17 (proportional)
          [80]    - is current number stuck? (0 or 1)
          [81]    - empty slots below current number / 18
          [82]    - empty slots above current number / 18
        """
        obs = [0.0] * 83

        # Slots (normalized)
        for i in range(SLOT_COUNT):
            obs[i] = self.slots[i] / SLOT_MAX

        # Current state
        obs[18] = self.number / SLOT_MAX
        obs[19] = self.next_number / SLOT_MAX
        obs[20] = self.level / SLOT_COUNT

        # Slot occupancy flags
        for i in range(SLOT_COUNT):
            obs[21 + i] = 1.0 if self.slots[i] != 0 else 0.0

        # Selectable powers (one-hot per slot)
        for i, p in enumerate(self.selectable_powers[:2]):
            if 1 <= p <= POWER_COUNT:
                obs[39 + i * POWER_COUNT + (p - 1)] = 1.0

        # Enabled powers (one-hot per slot, only if enabled)
        for i in range(min(3, len(self.selected_powers))):
            if i in self.enabled_powers:
                p = self.selected_powers[i]
                if 1 <= p <= POWER_COUNT:
                    obs[53 + i * POWER_COUNT + (p - 1)] = 1.0

        # Derived features
        valid_cur = self._valid_slots_for(self.number)
        valid_next = self._valid_slots_for(self.next_number)
        obs[74] = len(valid_cur) / SLOT_COUNT
        obs[75] = len(valid_next) / SLOT_COUNT

        # Min gap between adjacent filled numbers
        filled = [s for s in self.slots if s != 0]
        filled.sort()
        if len(filled) >= 2:
            min_gap = min(filled[i+1] - filled[i] for i in range(len(filled) - 1))
            obs[76] = min_gap / SLOT_MAX
        else:
            obs[76] = 1.0  # no gap constraint yet

        # Fraction filled
        obs[77] = len(filled) / SLOT_COUNT

        # Number of enabled powers
        obs[78] = len(self.enabled_powers) / 3.0

        # Ideal proportional position for current number
        obs[79] = (self.number - SLOT_MIN) / (SLOT_MAX - SLOT_MIN)

        # Is stuck?
        obs[80] = 0.0 if valid_cur else 1.0

        # Empty slots in valid range below/above current number
        empty_below = sum(1 for i in range(SLOT_COUNT) if self.slots[i] == 0 and i in valid_cur and self.number > 0)
        empty_above = sum(1 for i in range(SLOT_COUNT) if self.slots[i] == 0 and i in valid_cur)
        # More useful: slots below vs above the ideal position
        ideal_pos = (self.number - SLOT_MIN) / (SLOT_MAX - SLOT_MIN) * (SLOT_COUNT - 1)
        below = sum(1 for s in valid_cur if s < ideal_pos)
        above = sum(1 for s in valid_cur if s >= ideal_pos)
        obs[81] = below / max(SLOT_COUNT, 1)
        obs[82] = above / max(SLOT_COUNT, 1)

        return obs

    def board_quality(self) -> float:
        """Compute a 0-1 score of how good the current board state is.

        Higher = more future flexibility. Used for reward shaping.
        """
        valid_cur = len(self._valid_slots_for(self.number))
        valid_next = len(self._valid_slots_for(self.next_number))
        empty = sum(1 for s in self.slots if s == 0)

        if empty == 0:
            return 1.0  # board is full, game won

        # How many slots can current/next number use (relative to empty slots)
        flexibility = (valid_cur + valid_next) / (2 * max(empty, 1))

        # Min gap between adjacent filled numbers (tightness penalty)
        filled = sorted(s for s in self.slots if s != 0)
        if len(filled) >= 2:
            min_gap = min(filled[i+1] - filled[i] for i in range(len(filled) - 1))
            gap_score = min(min_gap / 50.0, 1.0)  # 50+ gap is healthy
        else:
            gap_score = 1.0

        return 0.6 * flexibility + 0.4 * gap_score

    # ------------------------------------------------------------------
    # Internal: number generation
    # ------------------------------------------------------------------

    def _draw_unique(self, exclude: list[int]) -> int:
        """Draw a random number in [SLOT_MIN, SLOT_MAX] not in exclude or current slots."""
        used = set(exclude)
        for s in self.slots:
            if s != 0:
                used.add(s)

        while True:
            n = self.rng.randint(SLOT_MIN, SLOT_MAX)
            if n not in used:
                return n

    def _draw_between(self, lo: int, hi: int) -> int:
        """Draw random in [lo, hi] inclusive."""
        if lo > hi:
            lo, hi = hi, lo
        if lo == hi:
            return lo
        return self.rng.randint(lo, hi)

    # ------------------------------------------------------------------
    # Internal: trap generation
    # ------------------------------------------------------------------

    def _generate_traps(self) -> list[int]:
        """Distribute 5 random traps across 18 slots.

        Uses the contract's probability-weighted algorithm:
        for each slot, probability = remaining_traps / remaining_slots.
        Trap types are drawn from a shuffled deck of [1..5] without replacement.
        """
        trap_types = list(range(1, TRAP_COUNT + 1))
        self.rng.shuffle(trap_types)
        deck_idx = 0

        traps = [0] * SLOT_COUNT
        remaining_traps = TRAP_COUNT
        remaining_slots = SLOT_COUNT

        for i in range(SLOT_COUNT):
            if remaining_traps <= 0:
                break
            # Probability check
            prob = remaining_traps / remaining_slots
            if self.rng.random() < prob:
                traps[i] = trap_types[deck_idx]
                deck_idx += 1
                remaining_traps -= 1
            remaining_slots -= 1

        return traps

    # ------------------------------------------------------------------
    # Internal: placement and traps
    # ------------------------------------------------------------------

    def _place_number(self, number: int, index: int) -> None:
        """Place a number at index, triggering any trap at that position."""
        assert self.slots[index] == 0, f"Slot {index} occupied by {self.slots[index]}"
        self.slots[index] = number

        # Check for trap
        trap_type = self.traps[index]
        if trap_type != TrapType.NONE and index not in self.disabled_traps:
            self.disabled_traps.add(index)
            self._apply_trap(trap_type, index)

    def _apply_trap(self, trap_type: int, slot_index: int) -> None:
        """Apply a trap effect at slot_index."""
        if trap_type == TrapType.BOMB:
            self._trap_bomb(slot_index)
        elif trap_type == TrapType.LUCKY:
            self._trap_lucky(slot_index)
        elif trap_type == TrapType.MAGNET:
            self._trap_magnet(slot_index)
        elif trap_type == TrapType.UFO:
            self._trap_ufo(slot_index)
        elif trap_type == TrapType.WINDY:
            self._trap_windy(slot_index)

    def _trap_bomb(self, slot_index: int) -> None:
        """Shuffle the 2 nearest filled neighbors within their outer boundaries."""
        trap_val = self.slots[slot_index]

        # Find nearest filled to the left + its outer boundary
        prev_idx = None
        prev_prev_val = SLOT_MIN
        prev_val = SLOT_MIN
        for j in range(slot_index - 1, -1, -1):
            if self.slots[j] != 0:
                if prev_idx is None:
                    prev_idx = j
                    prev_val = self.slots[j]
                else:
                    prev_prev_val = self.slots[j]
                    break

        # Find nearest filled to the right + its outer boundary
        next_idx = None
        next_next_val = SLOT_MAX
        next_val = SLOT_MAX
        for j in range(slot_index + 1, SLOT_COUNT):
            if self.slots[j] != 0:
                if next_idx is None:
                    next_idx = j
                    next_val = self.slots[j]
                else:
                    next_next_val = self.slots[j]
                    break

        # Shuffle neighbors
        if prev_idx is not None and prev_prev_val != 0:
            self.slots[prev_idx] = self._draw_between(prev_prev_val, trap_val)
        if next_idx is not None and next_next_val != 0:
            self.slots[next_idx] = self._draw_between(trap_val, next_next_val)

    def _trap_lucky(self, slot_index: int) -> None:
        """Shuffle the value at slot_index to random between its neighbors."""
        left_bound = SLOT_MIN
        for j in range(slot_index - 1, -1, -1):
            if self.slots[j] != 0:
                left_bound = self.slots[j]
                break

        right_bound = SLOT_MAX
        for j in range(slot_index + 1, SLOT_COUNT):
            if self.slots[j] != 0:
                right_bound = self.slots[j]
                break

        self.slots[slot_index] = self._draw_between(left_bound, right_bound)

    def _trap_magnet(self, slot_index: int) -> None:
        """Pull nearest non-adjacent numbers one slot closer."""
        # Left: find nearest filled that's not adjacent
        for j in range(slot_index - 1, -1, -1):
            if self.slots[j] != 0:
                if j + 1 == slot_index:
                    break  # already adjacent, skip
                # Move from j to j+1
                self._move(j, j + 1)
                break

        # Right: find nearest filled that's not adjacent
        for j in range(slot_index + 1, SLOT_COUNT):
            if self.slots[j] != 0:
                if j - 1 == slot_index:
                    break  # already adjacent, skip
                # Move from j to j-1
                self._move(j, j - 1)
                break

    def _trap_ufo(self, slot_index: int) -> None:
        """Move placed number to a random empty slot in the empty window around it."""
        # Find left boundary of empty window
        left_idx = slot_index
        for j in range(slot_index - 1, -1, -1):
            if self.slots[j] != 0:
                break
            left_idx = j

        # Find right boundary of empty window
        right_idx = slot_index
        for j in range(slot_index + 1, SLOT_COUNT):
            if self.slots[j] != 0:
                break
            right_idx = j

        # Move to a random position in [left_idx, right_idx]
        new_idx = self._draw_between(left_idx, right_idx)
        if new_idx != slot_index:
            self._move(slot_index, new_idx)

    def _trap_windy(self, slot_index: int) -> None:
        """Push nearest movable numbers one slot away."""
        # Left: find nearest filled with empty slot behind it
        for j in range(slot_index - 1, 0, -1):  # start from slot_index-1, stop at 1
            if self.slots[j] != 0:
                if self.slots[j - 1] == 0:
                    self._move(j, j - 1)
                break

        # Right: find nearest filled with empty slot ahead
        for j in range(slot_index + 1, SLOT_COUNT - 1):
            if self.slots[j] != 0:
                if self.slots[j + 1] == 0:
                    self._move(j, j + 1)
                break

    def _move(self, from_idx: int, to_idx: int) -> None:
        """Move a number from one slot to another (can trigger trap at destination)."""
        val = self.slots[from_idx]
        self.slots[from_idx] = 0
        self._place_number(val, to_idx)

    # ------------------------------------------------------------------
    # Internal: power effects
    # ------------------------------------------------------------------

    def _apply_power_effect(self, power_type: int) -> None:
        if power_type == PowerType.REROLL:
            self.number = self._draw_unique([])
        elif power_type == PowerType.HIGH:
            # Draw a number higher than current, fall back to full reroll
            used = set(s for s in self.slots if s != 0)
            used.add(self.number)
            candidates = [n for n in range(self.number, SLOT_MAX + 1) if n not in used]
            if candidates:
                self.number = self.rng.choice(candidates)
            else:
                self.number = self._draw_unique([])
        elif power_type == PowerType.LOW:
            # Draw a number lower than current, fall back to full reroll
            used = set(s for s in self.slots if s != 0)
            used.add(self.number)
            candidates = [n for n in range(SLOT_MIN, self.number + 1) if n not in used]
            if candidates:
                self.number = self.rng.choice(candidates)
            else:
                self.number = self._draw_unique([])

        elif power_type == PowerType.SWAP:
            self.number, self.next_number = self.next_number, self.number
        elif power_type == PowerType.DOUBLE_UP:
            self.number = min(self.number * 2, SLOT_MAX)
        elif power_type == PowerType.HALVE:
            self.number = max(self.number // 2, SLOT_MIN)
        elif power_type == PowerType.MIRROR:
            amplitude = SLOT_MAX + SLOT_MIN  # 1000
            if self.number > amplitude:
                self.number = SLOT_MIN
            else:
                self.number = amplitude - self.number

    # ------------------------------------------------------------------
    # Internal: game state updates
    # ------------------------------------------------------------------

    def _update(self) -> None:
        """Called after a successful placement. Level up, draw numbers, check over."""
        self.level += 1

        if not self.is_completed():
            # Shift numbers
            self.number = self.next_number
            used = [s for s in self.slots if s != 0]
            used.append(self.number)
            self.next_number = self._draw_unique(used)

        # Check game over
        self._check_game_over()

        # Draw powers if applicable
        if (
            not self.over
            and not self.is_completed()
            and self.level % DRAW_STAGE == 0
            and self.level < MAX_DRAW_LEVEL
            and not self.selectable_powers
        ):
            self.selectable_powers = self._draw_powers()

    def _draw_powers(self) -> list[int]:
        """Draw 2 distinct power types from [1..7]."""
        return self.rng.sample(range(1, POWER_COUNT + 1), DRAW_COUNT)

    def _check_game_over(self) -> None:
        """Check if game is over: can't place AND no selectable AND no enabled powers."""
        if self.is_completed() and not self.enabled_powers:
            self.over = True
            return

        can_place = bool(self._valid_slots_for(self.number))
        if not can_place and not self.selectable_powers and not self.enabled_powers:
            self.over = True

    # ------------------------------------------------------------------
    # Internal: validation
    # ------------------------------------------------------------------

    def _valid_slots_for(self, number: int) -> list[int]:
        """Find empty slots where number can be placed maintaining ascending order."""
        valid = []
        for i in range(SLOT_COUNT):
            if self.slots[i] != 0:
                continue

            # Find lower bound (nearest filled to the left)
            lower = 0
            for j in range(i - 1, -1, -1):
                if self.slots[j] != 0:
                    lower = self.slots[j]
                    break

            # Find upper bound (nearest filled to the right)
            upper = SLOT_MAX + 1  # effectively infinity
            for j in range(i + 1, SLOT_COUNT):
                if self.slots[j] != 0:
                    upper = self.slots[j]
                    break

            if number > lower and number < upper:
                valid.append(i)

        return valid

    def _is_valid(self) -> bool:
        """Check slots are in ascending order (ignoring zeros)."""
        prev = 0
        for i in range(SLOT_COUNT):
            if self.slots[i] == 0:
                continue
            if self.slots[i] < prev:
                return False
            prev = self.slots[i]
        return True


# ------------------------------------------------------------------
# Baseline bot (port of Rust bot logic)
# ------------------------------------------------------------------

def baseline_decide_slot(game: NumsGame) -> int:
    """Proportional mapping + next_number tiebreaker (matches Rust bot)."""
    valid = game.valid_slots()
    if not valid:
        return -1

    number = game.number
    ideal = round((number - SLOT_MIN) / (SLOT_MAX - SLOT_MIN) * (SLOT_COUNT - 1))

    # Find closest valid slot to ideal
    best = valid[0]
    best_dist = abs(valid[0] - ideal)
    for s in valid[1:]:
        d = abs(s - ideal)
        if d < best_dist:
            best = s
            best_dist = d

    # Tiebreaker: avoid blocking next_number
    test_slots = game.slots[:]
    test_slots[best] = number
    next_valid = _count_valid(test_slots, game.next_number)
    if next_valid == 0:
        for s in valid:
            if s == best:
                continue
            ts = game.slots[:]
            ts[s] = number
            if _count_valid(ts, game.next_number) > 0:
                return s

    return best


def baseline_select_power(game: NumsGame) -> int:
    """Fixed priority power selection (matches Rust bot)."""
    valid = game.valid_slots()
    is_stuck = len(valid) == 0

    if is_stuck:
        priority = [PowerType.REROLL, PowerType.SWAP, PowerType.MIRROR,
                     PowerType.HIGH, PowerType.LOW, PowerType.HALVE, PowerType.DOUBLE_UP]
    else:
        priority = [PowerType.SWAP, PowerType.REROLL, PowerType.MIRROR,
                     PowerType.HIGH, PowerType.LOW, PowerType.DOUBLE_UP, PowerType.HALVE]

    for target in priority:
        for i, p in enumerate(game.selectable_powers):
            if p == target:
                return i
    return 0


def baseline_apply_power(game: NumsGame) -> int:
    """Fixed priority power application (matches Rust bot)."""
    priority = [PowerType.REROLL, PowerType.SWAP, PowerType.MIRROR,
                PowerType.HALVE, PowerType.HIGH, PowerType.LOW, PowerType.DOUBLE_UP]

    enabled_list = sorted(game.enabled_powers)
    for target in priority:
        for idx in enabled_list:
            if idx < len(game.selected_powers) and game.selected_powers[idx] == target:
                return idx

    return enabled_list[0] if enabled_list else 0


def _count_valid(slots: list[int], number: int) -> int:
    """Count valid placements for a number in a slot configuration."""
    count = 0
    for i in range(len(slots)):
        if slots[i] != 0:
            continue
        lower = 0
        for j in range(i - 1, -1, -1):
            if slots[j] != 0:
                lower = slots[j]
                break
        upper = SLOT_MAX + 1
        for j in range(i + 1, len(slots)):
            if slots[j] != 0:
                upper = slots[j]
                break
        if number > lower and number < upper:
            count += 1
    return count


def play_baseline_game(seed: Optional[int] = None) -> int:
    """Play one full game using the baseline strategy. Returns final level."""
    game = NumsGame()
    game.reset(seed)

    while not game.over:
        phase = game.get_phase()
        if phase == "select":
            idx = baseline_select_power(game)
            game.select_power(idx)
        elif phase == "place":
            slot = baseline_decide_slot(game)
            game.place(slot)
        elif phase == "apply":
            idx = baseline_apply_power(game)
            game.apply_power(idx)
        else:
            break

    return game.level


if __name__ == "__main__":
    # Quick benchmark of baseline
    import time
    N = 10000
    t0 = time.time()
    levels = [play_baseline_game(seed=i) for i in range(N)]
    elapsed = time.time() - t0
    avg = sum(levels) / len(levels)
    best = max(levels)
    dist = {}
    for lv in levels:
        dist[lv] = dist.get(lv, 0) + 1
    print(f"Baseline over {N} games ({elapsed:.1f}s):")
    print(f"  Average level: {avg:.2f}")
    print(f"  Best level:    {best}")
    print(f"  Distribution:  {dict(sorted(dist.items()))}")
