"""
Advanced Gridworld for Gymnasium
--------------------------------
A configurable Gridworld environment compatible with Gymnasium (step API v0.28+). Designed to stress-test RL agents.

Key features
============
- Configurable grid size and walls density
- Static and moving obstacles (optionally adversarial)
- 3-slot inventory (Tile-coded) for picking up items that can used on obstacles
- Global wind and terrain-based slip probability (around water)
- Partial observability with egocentric crop & optional one-hot channels
- Multiple observation modes: "rgb", "tiles", "dict" (where "dict" = tiles + inventory + time)
- Tile channels: "onehot", "index", or "rgb"
- Render modes: "ansi", "rgb_array"
- Deterministic seeding and episode-level domain randomization
"""
import enum
from contextlib import suppress

import gymnasium as gym
import numpy as np
from dataclasses import dataclass
from gymnasium import spaces


# ----------------------------
# Action encoding
# ----------------------------

class Act(enum.IntEnum):
    # Movement / stay (unchanged indices 0..4 to preserve prior behavior)
    STAY = 0
    MOVE_RIGHT = 1
    MOVE_LEFT = 2
    MOVE_DOWN = 3
    MOVE_UP = 4

    # Recover from fallen state
    STAND_UP = 5

    # Use inventory slot s on adjacent tile dir (s in {1,2,3})
    USE1_LEFT = 6
    USE1_RIGHT = 7
    USE1_UP = 8
    USE1_DOWN = 9

    USE2_LEFT = 10
    USE2_RIGHT = 11
    USE2_UP = 12
    USE2_DOWN = 13

    USE3_LEFT = 14
    USE3_RIGHT = 15
    USE3_UP = 16
    USE3_DOWN = 17

    # Drop / switch at current tile, for slot s
    SWAP1 = 18
    SWAP2 = 19
    SWAP3 = 20


# ----------------------------
# Tile encoding & color table
# ----------------------------

class Tile(enum.IntEnum):
    AGENT = 0  # used only in rendering / observation composition
    APPLE = 1  # dense reward, consumed when moved over
    AXE = 2  #  not consumed, can be picked up, can be used to remove trees and moving obstacles
    BUCKET = 3  # not consumed, can be picked up, allows picking up water if held in inventory
    DOOR = 4  # stationary obstacle that can be removed with keys
    EMPTY = 5
    GOAL = 6  # sparse reward
    KEY = 7  # can be picked up, consumed to remove locked doors
    LAVA = 8  # stationary obstacle that can be removed with water
    MOVING_OBS = 9  # mobile obstacle that consumes apples and can be adversarial but can be removed with an axe
    PICK = 10  # not consumed, can be picked up, can be used to remove non-border walls
    TREE = 11  # stationary obstacle that can be removed with an axe but can drop apples, especially if watered
    WALL = 12
    WATER = 13  # chance to slip when moved over, can be picked up, consumed when used to remove lava or on trees

# RGB palette (uint8)
PALETTE = np.array([
    [ 80,  80, 255],  # AGENT
    [255, 215,   0],  # APPLE
    [160,  82,  45],  # AXE
    [135, 206, 250],  # BUCKET
    [150,  75,   0],  # DOOR
    [240, 240, 240],  # EMPTY
    [ 50, 180,  60],  # GOAL
    [ 30, 144, 255],  # KEY
    [220,  60,  30],  # LAVA
    [128,   0, 128],  # MOVING_OBS
    [205, 133,  63],  # PICK
    [ 34, 139,  34],  # TREE
    [ 40,  40,  40],  # WALL
    [ 64, 164, 223],  # WATER
], dtype=np.uint8)


# ----------------------------
# Configuration dataclass
# ----------------------------

@dataclass
class GridworldConfig:
    adversary: bool = False  # if True, one obstacle chases the agent
    apple_reward: float = 0.1
    apples: int = 0
    axes: int = 0
    buckets: int = 0
    channels: str = "onehot"  # "onehot" | "index" | "rgb"
    collision_penalty: float = -0.2
    domain_randomization: bool = True
    door_reward: float = 0.0
    doors: int = 0
    dynamic_obstacles: int = 0
    goal_respawn: bool = False
    goal_reward: float = 1.0
    goal_terminates: bool = True
    goals: int = 1
    height: int = 10
    keys: int = 0
    lava_penalty: float = -1.0
    lava_pools: int = 0
    max_steps: int = 200
    observation_mode: str = "dict"  # "dict" | "tiles" | "rgb"
    picks: int = 0
    render_mode: str | None = None  # "ansi" | "rgb_array"
    respawn_on_lava: bool = False
    seed: int | None = None
    slip_prob: float = 0.0
    step_penalty: float = 0.0
    tree_drop_chance: float = 0.01  # ambient per-step chance per tree
    tree_drop_chance_watered: float = 0.75  # one-shot boosted chance on watering
    trees: int = 0
    truncate_on_collision: bool = False
    view_size: int = 5  # odd -> egocentric crop size (POMDP); <=0 => full obs
    walls_density: float = 0.0  # random walls fraction (excluding borders)
    water_pools: int = 0
    width: int = 10
    wind_dir: tuple[int, int] = (0, 0)  # (dx, dy)
    wind_strength: float = 0.0  # probability push in wind_dir

    def validate(self) -> None:
        assert self.width >= 4 and self.height >= 4, "grid too small"
        assert self.view_size % 2 == 1 or self.view_size <= 0, "view_size must be odd or <=0"
        assert self.channels in {"onehot", "index", "rgb"}
        assert self.observation_mode in {"dict", "tiles", "rgb"}
        assert 0.0 <= self.slip_prob < 1.0
        assert 0.0 <= self.wind_strength < 1.0
        assert -1 <= self.wind_dir[0] <= 1 and -1 <= self.wind_dir[1] <= 1, "wind_dir components must be in {-1,0,1}"
        assert 0.0 <= self.walls_density <= 1.0, "walls_density must be in [0,1]"
        assert self.max_steps > 0, "max_steps must be positive"
        # counts
        for name in ("goals", "keys", "doors", "apples", "axes", "buckets",
                     "lava_pools", "trees", "water_pools", "dynamic_obstacles"):
            assert getattr(self, name) >= 0, f"{name} must be >= 0"

# ----------------------------
# Environment
# ----------------------------

class AdvancedGridworldEnv(gym.Env):
    metadata = {"render_modes": ["ansi", "rgb_array"], "render_fps": 8}

    MOVE_DELTAS = {
        Act.STAY: (0, 0),
        Act.MOVE_RIGHT: (1, 0),
        Act.MOVE_LEFT: (-1, 0),
        Act.MOVE_DOWN: (0, 1),
        Act.MOVE_UP: (0, -1),
    }

    def __init__(self, config: GridworldConfig | None = None):
        super().__init__()
        self.config = config or GridworldConfig()
        self.config.validate()

        self._dynamic_obs_set: set[tuple[int, int]] = set()
        self._rng = np.random.default_rng(self.config.seed)
        self._slip_ep = self.config.slip_prob
        self._wind_dir_ep: tuple[int, int] = self.config.wind_dir
        self._wind_strength_ep: float = self.config.wind_strength
        self.agent_pos = (1, 1)
        self.apple_positions: list[tuple[int, int]] = []
        self.door_positions: list[tuple[int, int]] = []
        self.dynamic_obs: list[tuple[int, int]] = []
        self.fallen_down: bool = False
        self.goal_positions: list[tuple[int, int]] = []
        self.grid = np.zeros((self.config.height, self.config.width), dtype=np.int8)
        self.inventory_slots: list[int] = [int(Tile.EMPTY), int(Tile.EMPTY), int(Tile.EMPTY)]
        self.key_positions: list[tuple[int, int]] = []
        self.step_count = 0
        self.tree_positions: list[tuple[int, int]] = []

        # Spaces
        self.action_space = spaces.Discrete(len(Act))
        obs_space = self._make_observation_space()
        if self.config.observation_mode == "dict":
            self.observation_space = spaces.Dict({
                "tiles": obs_space,
                "inventory": spaces.Box(low=0, high=len(Tile) - 1, shape=(3,), dtype=np.uint8),
                "time": spaces.Box(low=0.0, high=1.0, shape=(1, ), dtype=np.float32),
                "status": spaces.MultiBinary(1),
            })
        else:
            self.observation_space = obs_space

    # ------------- Gym API -------------

    def close(self):
        # clear references to help GC in long runs.
        self.dynamic_obs = []
        self._dynamic_obs_set = set()
        with suppress(Exception):
            super().close()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        eff_seed = self.config.seed if seed is None else seed
        # Let Gymnasium handle np_random seeding and bookkeeping.
        super().reset(seed=eff_seed)
        # Use Gym's episode-scoped Generator directly; single source of randomness.
        self._rng = self.np_random

        self.fallen_down = False
        self.inventory_slots = [int(Tile.EMPTY), int(Tile.EMPTY), int(Tile.EMPTY)]
        self.step_count = 0

        self._generate_layout()
        self._slip_ep, self._wind_dir_ep, self._wind_strength_ep = self._episode_params()
        obs = self._get_observation()
        info = {"agent_pos": self.agent_pos}
        return obs, info

    def step(self, action: int):
        self.step_count += 1
        acted_nonmove = False  # True if we executed a use/swap action (consumes step)
        reward = 0.0
        terminated = False
        truncated = False

        # Remember the agent’s previous position and fallen state
        prev_agent_pos = self.agent_pos
        prev_fallen = self.fallen_down

        # Branch on action family
        try:
            act_enum = Act(int(action))
        except (ValueError, TypeError):
            act_enum = Act.STAY

        slipped_this_step = False  # did we newly fall this step?

        if act_enum in self.MOVE_DELTAS:
            # Movement / stay
            if self.fallen_down:
                # When down, movement actions do nothing.
                new_pos = self.agent_pos
            else:
                dx, dy = self.MOVE_DELTAS[act_enum]
                proposed = (self.agent_pos[0] + dx, self.agent_pos[1] + dy)
                proposed = self._clip_to_bounds(proposed)

                # Slip is only possible when moving into or out of water
                src_is_water = (self._tile_at(self.agent_pos) == Tile.WATER)
                dst_is_water = (self._tile_at(proposed) == Tile.WATER)
                if (src_is_water or dst_is_water) and (self._rng.random() < self._slip_ep):
                    # Slip: agent falls down instead of moving
                    self.fallen_down = True
                    slipped_this_step = True
                    proposed = self.agent_pos  # stay in place

                # Wind: probabilistic push (as a separate single-cell attempt)
                if (not slipped_this_step) and self._wind_strength_ep > 0 and self._rng.random() < self._wind_strength_ep:
                    wdx, wdy = self._wind_dir_ep
                    wx, wy = int(np.sign(wdx)), int(np.sign(wdy))
                    cand = self._clip_to_bounds((proposed[0] + wx, proposed[1] + wy))
                    proposed = cand

                new_pos = proposed

        else:
            # Non-movement actions consume the step but do not move the agent
            acted_nonmove = True
            new_pos = self.agent_pos

            if act_enum == Act.STAND_UP:
                # Recover from fallen state
                self.fallen_down = False
            else:
                # Use slot on adjacent tile?
                sd = self._use_action_to_slot_dir(action)
                if sd is not None:
                    s_idx, (ux, uy) = sd
                    reward += self._use_slot_on_dir(s_idx, ux, uy)

                # Swap/Drop at current tile?
                elif act_enum in (Act.SWAP1, Act.SWAP2, Act.SWAP3):
                    s_idx = {Act.SWAP1: 0, Act.SWAP2: 1, Act.SWAP3: 2}[act_enum]
                    reward += self._swap_drop_pick_at_current(s_idx)

                # Unknown action types fall through as no-ops

        # Collision logic
        apples_eaten_by_obs = 0
        collided = False
        pushed = False
        t = self._tile_at(new_pos)
        if t in (Tile.WALL, Tile.DOOR, Tile.TREE):
            collided = True
            reward += self.config.collision_penalty
            if self.config.truncate_on_collision:
                truncated = True
            # stay in place
            new_pos = self.agent_pos
        elif t == Tile.LAVA:
            reward += self.config.lava_penalty
            if self.config.respawn_on_lava:
                new_pos = self._random_empty_cell(exclude=self._dynamic_obs_set)
            else:
                terminated = True
        else:
            # collect tokens / interact
            if t == Tile.APPLE:
                reward += self.config.apple_reward
                if new_pos in self.apple_positions:
                    self.apple_positions.remove(new_pos)
                self.grid[new_pos[1], new_pos[0]] = int(Tile.EMPTY)
            elif t == Tile.GOAL:
                reward += self.config.goal_reward
                # remove from bookkeeping + clear cell
                self._remove_goal_at(new_pos)
                self.grid[new_pos[1], new_pos[0]] = int(Tile.EMPTY)
                terminated = bool(self.config.goal_terminates)
                if self.config.goal_respawn and not terminated:
                    self._spawn_goal(1)

            # obstacle-at-destination check on the layered field
            if new_pos in self._dynamic_obs_set:
                collided = True
                reward += self.config.collision_penalty
                back = prev_agent_pos

                if self.config.truncate_on_collision:
                    truncated = True
                    new_pos = self.agent_pos
                else:
                    new_pos = back if self._agent_can_occupy(back) else self.agent_pos

        # Update to new position
        self.agent_pos = new_pos

        if not terminated:
            # Remember previous obstacle positions, then move dynamic obstacles
            prev_obs_positions = list(self.dynamic_obs)
            apples_eaten_by_obs = self._step_dynamic_obstacles()

            # After obstacles move, if one occupies the agent cell, push the agent
            if self.agent_pos in self.dynamic_obs:
                reward += self.config.collision_penalty
                pushed = True

                # Find (one) colliding obstacle and its movement vector
                coll_idx = None
                for i, pos in enumerate(self.dynamic_obs):
                    if pos == self.agent_pos:
                        coll_idx = i
                        break

                if coll_idx is not None:
                    ox_prev = prev_obs_positions[coll_idx][0]
                    oy_prev = prev_obs_positions[coll_idx][1]
                    ox_now, oy_now = self.dynamic_obs[coll_idx]
                    odx = ox_now - ox_prev
                    ody = oy_now - oy_prev

                    # Default candidate: push along obstacle's move direction
                    if odx == 0 and ody == 0:
                        # Obstacle didn't move; fall back to pushing the agent back to where it came from
                        cand = prev_agent_pos
                    else:
                        cand = (self.agent_pos[0] + odx, self.agent_pos[1] + ody)

                    # Try push destination; if not safe, revert to previous cell if safe; else stay
                    if self._agent_can_occupy(cand):
                        self.agent_pos = cand
                    elif self._agent_can_occupy(prev_agent_pos):
                        self.agent_pos = prev_agent_pos
                    else:
                        # Last-resort escape to avoid persistent co-location.
                        # Fixed neighbor order for determinism:
                        for nx, ny in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            cand2 = (self.agent_pos[0] + nx, self.agent_pos[1] + ny)
                            if self._agent_can_occupy(cand2):
                                self.agent_pos = cand2
                                break

                still_colliding = (self.agent_pos in self._dynamic_obs_set)
                # Only truncate if collision remains unresolved after push
                if self.config.truncate_on_collision and still_colliding:
                    truncated = True

            # Trees may drop apples with each step
            self._trees_maybe_drop_apples()

            # If a push moves the agent onto an apple/key/goal, process the landing tile
            t_post = self._tile_at(self.agent_pos)
            if t_post == Tile.APPLE:
                reward += self.config.apple_reward
                if self.agent_pos in self.apple_positions:
                    self.apple_positions.remove(self.agent_pos)
                self.grid[self.agent_pos[1], self.agent_pos[0]] = int(Tile.EMPTY)
            elif t_post == Tile.GOAL:
                reward += self.config.goal_reward
                # remove from bookkeeping + clear cell
                self._remove_goal_at(self.agent_pos)
                self.grid[self.agent_pos[1], self.agent_pos[0]] = int(Tile.EMPTY)
                terminated = bool(self.config.goal_terminates)
                if self.config.goal_respawn and not terminated:
                    self._spawn_goal(1)

            # Step penalty
            reward += -abs(self.config.step_penalty)

            # Time truncation
            if self.step_count >= self.config.max_steps:
                truncated = True

        obs = self._get_observation()
        info = {
            "acted_nonmove": acted_nonmove,
            "action": int(act_enum),
            "agent_pos": self.agent_pos,
            "apples_eaten_by_obs": apples_eaten_by_obs,
            "collided": collided or pushed,
            "dst_is_water": (self._tile_at(self.agent_pos) == Tile.WATER),
            "fallen_down": self.fallen_down,
            "fell_this_step": slipped_this_step,
            "previously_fallen": prev_fallen,
            "slip_prob_ep": self._slip_ep,
            "src_is_water": (self._tile_at(prev_agent_pos) == Tile.WATER),
            "step": self.step_count,
            "wind_dir_ep": self._wind_dir_ep,
            "wind_strength_ep": self._wind_strength_ep,
        }
        return obs, float(reward), bool(terminated), bool(truncated), info

    # ------------- Observation helpers -------------

    def _get_observation(self):
        grid = self.grid.copy()
        for (ox, oy) in self.dynamic_obs:  # overlay dynamic obstacles first
            if 0 <= ox < self.config.width and 0 <= oy < self.config.height:
                grid[oy, ox] = int(Tile.MOVING_OBS)
        ax, ay = self.agent_pos
        grid[ay, ax] = int(Tile.AGENT)

        if self.config.view_size > 0:
            v = self.config.view_size
            r = v // 2
            x0, x1 = ax - r, ax + r + 1
            y0, y1 = ay - r, ay + r + 1
            # pad with walls outside bounds for consistent POMDP
            padded = np.pad(
                grid, pad_width=r, mode="constant", constant_values=int(Tile.WALL)
            )
            crop = padded[y0 + r:y1 + r, x0 + r:x1 + r]
        else:
            crop = grid

        if self.config.observation_mode == "rgb":
            rgb = PALETTE[np.clip(crop, 0, len(PALETTE)-1)]
            return rgb

        if self.config.channels == "onehot":
            C = len(Tile)
            obs_tiles = np.eye(C, dtype=np.uint8)[np.clip(crop, 0, C-1)]
        elif self.config.channels == "rgb":
            obs_tiles = PALETTE[np.clip(crop, 0, len(PALETTE)-1)]
        else:  # index
            obs_tiles = crop.astype(np.uint8)

        if self.config.observation_mode == "tiles":
            return obs_tiles
        elif self.config.observation_mode == "dict":
            tfrac = np.array([min(self.step_count, self.config.max_steps) / max(1, self.config.max_steps)], dtype=np.float32)
            return {
                "tiles": obs_tiles,
                "inventory": np.array(self.inventory_slots, dtype=np.uint8),
                "time": tfrac,
                "status": np.array([int(self.fallen_down)], dtype=np.int8),
            }
        else:  # fallback
            return obs_tiles

    def _make_observation_space(self) -> spaces.Space:
        H, W = self.config.height, self.config.width
        if self.config.view_size > 0:
            V = self.config.view_size
            obs_shape = (V, V)
        else:
            obs_shape = (H, W)

        if self.config.observation_mode == "rgb":
            shape = obs_shape + (3,)
            return spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)

        if self.config.channels == "onehot":
            C = len(Tile)
            shape = obs_shape + (C,)
            return spaces.Box(low=0, high=1, shape=shape, dtype=np.uint8)

        if self.config.channels == "rgb":
            shape = obs_shape + (3,)
            return spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)

        # index-coded tiles
        return spaces.Box(low=0, high=len(Tile) - 1, shape=obs_shape, dtype=np.uint8)

    # ------------- Layout generation -------------

    def _generate_layout(self):
        H, W = self.config.height, self.config.width
        self.grid[:, :] = int(Tile.EMPTY)

        # Border walls
        self.grid[0, :] = int(Tile.WALL)
        self.grid[H-1, :] = int(Tile.WALL)
        self.grid[:, 0] = int(Tile.WALL)
        self.grid[:, W-1] = int(Tile.WALL)

        # Random internal walls
        if self.config.walls_density > 0:
            num_cells = (H-2) * (W-2)
            k = int(self.config.walls_density * num_cells)
            idx = self._rng.choice(num_cells, size=k, replace=False)
            ys = 1 + (idx // (W-2))
            xs = 1 + (idx % (W-2))
            self.grid[ys, xs] = int(Tile.WALL)

        # Doors
        self.door_positions = []
        for _ in range(max(0, self.config.doors)):
            pos = self._random_empty_cell()
            self.grid[pos[1], pos[0]] = int(Tile.DOOR)
            self.door_positions.append(pos)

        # Lava pools
        for _ in range(max(0, self.config.lava_pools)):
            pos = self._random_empty_cell()
            self.grid[pos[1], pos[0]] = int(Tile.LAVA)

        # Trees
        self.tree_positions = []
        for _ in range(max(0, self.config.trees)):
            pos = self._random_empty_cell()
            self.grid[pos[1], pos[0]] = int(Tile.TREE)
            self.tree_positions.append(pos)

        # Goals
        self.goal_positions = []
        self._spawn_goal(max(0, self.config.goals))

        # Apple
        self.apple_positions = []
        for _ in range(max(0, self.config.apples)):
            pos = self._random_empty_cell()
            self.grid[pos[1], pos[0]] = int(Tile.APPLE)
            self.apple_positions.append(pos)

        # Axes
        for _ in range(max(0, self.config.axes)):
            pos = self._random_empty_cell()
            self.grid[pos[1], pos[0]] = int(Tile.AXE)

        # Buckets
        for _ in range(max(0, self.config.buckets)):
            pos = self._random_empty_cell()
            self.grid[pos[1], pos[0]] = int(Tile.BUCKET)

        # Keys
        self.key_positions = []
        for _ in range(max(0, self.config.keys)):
            pos = self._random_empty_cell()
            self.grid[pos[1], pos[0]] = int(Tile.KEY)
            self.key_positions.append(pos)

        # Picks
        for _ in range(max(0, self.config.picks)):
            pos = self._random_empty_cell()
            self.grid[pos[1], pos[0]] = int(Tile.PICK)

        # Water pools
        for _ in range(max(0, self.config.water_pools)):
            pos = self._random_empty_cell()
            self.grid[pos[1], pos[0]] = int(Tile.WATER)

        if not np.any(self.grid == int(Tile.EMPTY)):
            raise RuntimeError(
                "Layout contains no EMPTY cell (try lowering walls_density or counts)."
            )

        # Dynamic obstacles (layered; do not write into base grid)
        self.dynamic_obs = []
        taken = set()
        for _ in range(max(0, self.config.dynamic_obstacles)):
            pos = self._random_empty_cell(exclude=taken)
            self.dynamic_obs.append(pos)
            taken.add(pos)
        self._dynamic_obs_set = set(self.dynamic_obs)

        # Agent spawn (avoid hazards)
        self.agent_pos = self._random_empty_cell(exclude=self._dynamic_obs_set)

    def _is_border(self, x: int, y: int) -> bool:
        return y == 0 or y == (self.config.height - 1) or x == 0 or x == (self.config.width - 1)

    def _remove_goal_at(self, pos: tuple[int, int]) -> None:
        """Remove goal bookkeeping for a goal at pos, if present."""
        try:
            self.goal_positions.remove(pos)
        except ValueError:
            pass

    def _spawn_goal(self, n: int):
        # Avoid spawning on the agent, on obstacles, or on existing goals.
        exclude = {self.agent_pos} | set(self.dynamic_obs) | set(self.goal_positions)
        for _ in range(n):
            pos = self._random_empty_cell(exclude=exclude)
            self.grid[pos[1], pos[0]] = int(Tile.GOAL)
            self.goal_positions.append(pos)
            exclude.add(pos)  # avoid placing multiple goals on same cell

    # ------------- Dynamics helpers -------------

    def _agent_can_occupy(self, p):
        """Is the target cell safe for the agent?"""
        x, y = p
        if not (0 <= x < self.config.width and 0 <= y < self.config.height):
            return False
        tile = Tile(int(self.grid[y, x]))
        if tile in (Tile.WALL, Tile.LAVA, Tile.DOOR, Tile.TREE):
            return False
        # can't move into another obstacle either
        if p in self.dynamic_obs and p != self.agent_pos:
            return False
        return True

    def _attempt_tree_drop_at(self, tree_pos: tuple[int, int], boosted: bool = False):
        """Attempt to drop an apple near a single tree position."""
        p = self.config.tree_drop_chance_watered if boosted else self.config.tree_drop_chance
        if self._rng.random() >= p:
            return
        empties = [(cx, cy) for (cx, cy) in self._neighbors4_tree_drop(tree_pos)
                   if self.grid[cy, cx] == int(Tile.EMPTY) and (cx, cy) not in self._dynamic_obs_set]
        if not empties:
            return
        drop = empties[self._rng.integers(0, len(empties))]
        # guards against double bookkeeping; self-heals stale apple_positions while avoiding duplicates
        if self.grid[drop[1], drop[0]] == int(Tile.EMPTY):
            self.grid[drop[1], drop[0]] = int(Tile.APPLE)
            if drop not in self.apple_positions:
                self.apple_positions.append(drop)

    def _clip_to_bounds(self, pos: tuple[int, int]) -> tuple[int, int]:
        x = int(np.clip(pos[0], 0, self.config.width - 1))
        y = int(np.clip(pos[1], 0, self.config.height - 1))
        return x, y

    def _episode_params(self):
        """Derive effective episode-specific params without mutating self.config."""
        slip = self.config.slip_prob
        wind_dir = self.config.wind_dir
        wind_strength = self.config.wind_strength
        if self.config.domain_randomization:
            if self.config.slip_prob > 0:
                slip = float(np.clip(self._rng.normal(self.config.slip_prob, 0.02), 0.0, 1.0 - 1e-6))
            if self.config.wind_strength > 0:
                wind_dir = tuple(int(x) for x in self._rng.choice([-1, 0, 1], size=2))
                # optional: a tiny jitter to strength as well (bounded)
                wind_strength = float(np.clip(
                    self._rng.normal(self.config.wind_strength, 0.02), 0.0, 1.0 - 1e-6))
        return slip, wind_dir, wind_strength

    def _neighbors4_tree_drop(self, pos: tuple[int, int]):
        x, y = pos
        cand = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        return [(cx, cy) for (cx, cy) in cand
                if 0 <= cx < self.config.width and 0 <= cy < self.config.height]

    def _random_empty_cell(self, exclude: set[tuple[int, int]] | None = None) -> tuple[int, int]:
        """Sample an EMPTY base-grid cell not in `exclude` (if given)."""
        if exclude is None:
            exclude = set()

        H, W = self.config.height, self.config.width

        # Fast rejection sampling (avoids materializing full list in the common case)
        for _ in range(5000):
            x = int(self._rng.integers(1, W - 1))
            y = int(self._rng.integers(1, H - 1))
            if self.grid[y, x] == int(Tile.EMPTY) and (x, y) not in exclude:
                return x, y

        # Deterministic fallback over the true set of empties
        empties = np.argwhere(self.grid == int(Tile.EMPTY))
        if empties.size == 0:
            raise RuntimeError("No EMPTY cells available in grid (layout over-constrained).")

        # Shuffle a copy to preserve RNG determinism without huge loops
        self._rng.shuffle(empties)
        for y, x in empties:
            p = (int(x), int(y))
            if p not in exclude:
                return p

        # If we got here, nothing fits the exclusion set.
        raise RuntimeError("No EMPTY cells remain after applying `exclude`.")

    def _step_dynamic_obstacles(self) -> int:
        """Advance obstacle layer via random walk (or adversary bias) without touching base tiles."""
        apples_eaten = 0
        next_taken: set[tuple[int, int]] = set()
        new_positions: list[tuple[int, int]] = []

        # Snapshot of current positions to disallow moves into other obstacles' current cells
        current_positions = set(self.dynamic_obs)

        for pos in list(self.dynamic_obs):  # iterate a snapshot
            if self.config.adversary:
                # biased move towards agent (Manhattan greedy)
                dx = np.sign(self.agent_pos[0] - pos[0])
                dy = np.sign(self.agent_pos[1] - pos[1])
                # Greedy first; stay is considered explicitly at the end
                candidate_steps = [(dx, 0), (0, dy), (-dx, 0), (0, -dy), (0, 0)]
            else:
                # random walk with stay
                candidate_steps = [(1, 0), (-1, 0), (0, 1), (0, -1), (0, 0)]
                self._rng.shuffle(candidate_steps)

            moved = None
            for (cx, cy) in candidate_steps:
                nx, ny = pos[0] + cx, pos[1] + cy
                # in-bounds
                if not (0 <= nx < self.config.width and 0 <= ny < self.config.height):
                    continue
                tile = Tile(int(self.grid[ny, nx]))
                # cannot pass through hard obstacles
                if tile in (Tile.WALL, Tile.LAVA, Tile.DOOR, Tile.TREE):
                    continue
                # do not move into a cell that another obstacle already reserved this tick
                if (nx, ny) in next_taken:
                    continue
                # do not move into any other obstacle's *current* cell
                if (nx, ny) in current_positions and (nx, ny) != pos:
                    continue
                moved = (nx, ny)
                break

            if moved is None:
                # staying in place is always allowed; if someone already moved into our cell,
                # the check above prevented that.
                moved = pos

            # If the obstacle's destination contains an apple, consume it.
            if self.grid[moved[1], moved[0]] == int(Tile.APPLE):
                # remove from bookkeeping list if present
                if moved in self.apple_positions:
                    self.apple_positions.remove(moved)
                # clear the base grid cell
                self.grid[moved[1], moved[0]] = int(Tile.EMPTY)
                apples_eaten += 1

            # update positions (avoid stacking handled above)
            next_taken.add(moved)
            new_positions.append(moved)

        self.dynamic_obs = new_positions
        self._dynamic_obs_set = set(new_positions)
        return apples_eaten

    def _tile_at(self, pos: tuple[int, int]) -> Tile:
        x, y = pos
        return Tile(int(self.grid[y, x]))

    def _trees_maybe_drop_apples(self):
        """Ambient apple-drops from all trees (low probability)."""
        if not self.tree_positions:
            return
        for pos in list(self.tree_positions):
            self._attempt_tree_drop_at(pos, boosted=False)

    # ------------- Inventory helpers -------------

    def _post_mutation_maybe_spill(self):
        if not any(x == int(Tile.BUCKET) for x in self.inventory_slots):
            # If no bucket remains but some WATER exists, spill them
            if any(x == int(Tile.WATER) for x in self.inventory_slots):
                self._spill_all_water()

    def _slot_empty(self, s: int) -> bool:
        return self.inventory_slots[s] == int(Tile.EMPTY)

    def _spill_all_water(self) -> None:
        """
        Spill (drop) all WATER units from inventory into the world when no bucket is held.
        Priority:
          1) Current tile if EMPTY
          2) First EMPTY 4-neighbor (fixed order for determinism)
          3) If nowhere to place, water is lost (evaporates)
        """
        if any(x == int(Tile.BUCKET) for x in self.inventory_slots):
            return  # still have a bucket; no spill

        ax, ay = self.agent_pos

        # Collect indices of WATER to remove
        water_slots = [i for i, it in enumerate(self.inventory_slots) if it == int(Tile.WATER)]
        for s in water_slots:
            # Try current tile
            if self.grid[ay, ax] == int(Tile.EMPTY):
                self.grid[ay, ax] = int(Tile.WATER)
            else:
                # Try neighbors in fixed order
                for nx, ny in ((ax + 1, ay), (ax - 1, ay), (ax, ay + 1), (ax, ay - 1)):
                    if 0 <= nx < self.config.width and 0 <= ny < self.config.height:
                        if self.grid[ny, nx] == int(Tile.EMPTY):
                            self.grid[ny, nx] = int(Tile.WATER)
                            break
                # If nowhere to place, we "evaporate" it (spill to nowhere)
            # Remove from inventory
            self.inventory_slots[s] = int(Tile.EMPTY)

    def _swap_drop_pick_at_current(self, s: int) -> float:
        """
        Swap/Drop/Pick at current tile:
          - If slot empty and tile holds a pickable item -> pick it.
          - If slot has item and tile empty -> drop it.
          - If both have items -> swap.
          - If tile holds non-pickable -> do nothing.
        Returns reward delta (0 here; extend if needed).
        """
        if s < 0 or s > 2:
            return 0.0
        ax, ay = self.agent_pos
        t = Tile(int(self.grid[ay, ax]))

        # Define pickable items for inventory
        pickable = {
            Tile.AXE,
            Tile.BUCKET,
            Tile.KEY,
            Tile.PICK,
        }

        # WATER is pickable only if BUCKET is present in inventory
        has_bucket = any(x == int(Tile.BUCKET) for x in self.inventory_slots)
        if has_bucket:
            pickable = pickable | {Tile.WATER}

        slot_empty = self._slot_empty(s)
        tile_has_item = t in pickable

        if slot_empty and tile_has_item:
            # pick up -> remove from grid
            self.inventory_slots[s] = int(t)
            self.grid[ay, ax] = int(Tile.EMPTY)
            # bookkeeping for tracked item types
            if t == Tile.KEY and (ax, ay) in self.key_positions:
                self.key_positions.remove((ax, ay))
            return 0.0

        if (not slot_empty) and t == Tile.EMPTY:
            # drop from slot to tile
            item = self._take_from_slot(s)
            self.grid[ay, ax] = int(item)
            if item == int(Tile.KEY) and (ax, ay) not in self.key_positions:
                self.key_positions.append((ax, ay))
            # If we dropped a bucket, we may need to spill water
            self._post_mutation_maybe_spill()
            return 0.0

        if (not slot_empty) and tile_has_item:
            # swap item in slot with tile item
            item_in_slot = self.inventory_slots[s]
            # place tile item into slot
            self.inventory_slots[s] = int(t)
            # place previous slot item onto tile
            self.grid[ay, ax] = int(item_in_slot)
            # update bookkeeping for tracked types
            if t == Tile.KEY and (ax, ay) in self.key_positions:
                self.key_positions.remove((ax, ay))
            if item_in_slot == int(Tile.KEY) and (ax, ay) not in self.key_positions:
                self.key_positions.append((ax, ay))
            # If swap removed the last bucket from inventory, spill water
            self._post_mutation_maybe_spill()
            return 0.0

        # Not pickable or no capacity at chosen slot
        return 0.0

    def _take_from_slot(self, s: int) -> int:
        """Remove and return tile code from slot s. Returns Tile.EMPTY if empty."""
        item = self.inventory_slots[s]
        self.inventory_slots[s] = int(Tile.EMPTY)
        return item

    @staticmethod
    def _use_action_to_slot_dir(action: int):
        mapping = {
            Act.USE1_LEFT: (0, (-1, 0)), Act.USE1_RIGHT: (0, (1, 0)),
            Act.USE1_UP: (0, (0, -1)), Act.USE1_DOWN: (0, (0, 1)),
            Act.USE2_LEFT: (1, (-1, 0)), Act.USE2_RIGHT: (1, (1, 0)),
            Act.USE2_UP: (1, (0, -1)), Act.USE2_DOWN: (1, (0, 1)),
            Act.USE3_LEFT: (2, (-1, 0)), Act.USE3_RIGHT: (2, (1, 0)),
            Act.USE3_UP: (2, (0, -1)), Act.USE3_DOWN: (2, (0, 1)),
        }
        return mapping.get(int(action))

    def _use_slot_on_dir(self, s: int, dx: int, dy: int) -> float:
        """
        Use item in slot s on adjacent tile (dx, dy). Returns reward delta.
        Implemented interactions:
          - KEY on DOOR: unlocks door (consumes key), gives door_reward.
          - AXE on TREE: chops tree (removes it).
          - AXE on MOVING_OBS (if present at target): removes that obstacle.
          - PICK on WALL: breaks wall (removes it).
          - WATER on LAVA: removes lava (consumes water).
          - WATER on TREE: triggers boosted apple drop near tree (consumes water).
        """
        if s < 0 or s > 2:
            return 0.0
        item = self.inventory_slots[s]
        if item == int(Tile.EMPTY):
            return 0.0

        ax, ay = self.agent_pos
        tx, ty = ax + dx, ay + dy
        if not (0 <= tx < self.config.width and 0 <= ty < self.config.height):
            return 0.0

        t = Tile(int(self.grid[ty, tx]))
        reward_delta = 0.0

        # AXE -> chop tree or remove moving obstacle at target cell
        if item == int(Tile.AXE):
            # Chop TREE
            if t == Tile.TREE:
                self.grid[ty, tx] = int(Tile.EMPTY)
                if (tx, ty) in self.tree_positions:
                    self.tree_positions.remove((tx, ty))
                return reward_delta
            # Remove MOVING_OBS at target location (layered)
            if (tx, ty) in self._dynamic_obs_set:
                self.dynamic_obs = [p for p in self.dynamic_obs if p != (tx, ty)]
                self._dynamic_obs_set.discard((tx, ty))
                if len(self._dynamic_obs_set) != len(self.dynamic_obs):
                    self._dynamic_obs_set = set(self.dynamic_obs)
                return reward_delta

        # KEY -> unlock adjacent locked door
        if item == int(Tile.KEY) and t == Tile.DOOR:
            self.grid[ty, tx] = int(Tile.EMPTY)
            self._take_from_slot(s)  # consume key
            if (tx, ty) in self.door_positions:
                self.door_positions.remove((tx, ty))
            reward_delta += float(self.config.door_reward)
            return reward_delta

        # PICKAXE -> remove non-border WALL
        if item == int(Tile.PICK) and t == Tile.WALL:
            if not self._is_border(tx, ty):
                self.grid[ty, tx] = int(Tile.EMPTY)
            return reward_delta

        # WATER -> extinguish lava or water a tree (boost apple drop)
        if item == int(Tile.WATER):
            # Extinguish LAVA
            if t == Tile.LAVA:
                self.grid[ty, tx] = int(Tile.EMPTY)
                self._take_from_slot(s)  # consume water
                return reward_delta
            # Water TREE: boosted apple drop attempt near that tree; tree remains
            if t == Tile.TREE:
                self._take_from_slot(s)  # consume water
                self._attempt_tree_drop_at((tx, ty), boosted=True)
                return reward_delta

        # Placeholder: extend for other item uses here
        return 0.0

    # ------------- Rendering -------------

    def render(self):
        mode = self.config.render_mode
        match mode:
            case 'ansi':
                return self._render_ansi()
            case 'rgb_array':
                return self._render_rgb()
            case _:
                raise RuntimeError("render_mode is None; set to 'ansi' or 'rgb_array'.")

    def _render_ansi(self) -> str:
        glyphs = {
            Tile.AGENT: " 😁 ",
            Tile.APPLE: " 🍎 ",
            Tile.AXE: " 🪓 ",
            Tile.BUCKET: " 🪣 ",
            Tile.DOOR: " 🚪 ",
            Tile.EMPTY: " ⬜ ",
            Tile.GOAL: " 🏁 ",
            Tile.KEY: " 🔑 ",
            Tile.LAVA: " ♨️ ",
            Tile.MOVING_OBS: " 😈 ",
            Tile.PICK: " ⛏️ ",
            Tile.TREE: " 🌳 ",
            Tile.WALL: " 🪨 ",
            Tile.WATER: " 💦 ",
        }
        grid = self.grid.copy()
        for (ox, oy) in self.dynamic_obs:  # overlay dynamic obstacles first
            if 0 <= ox < self.config.width and 0 <= oy < self.config.height:
                grid[oy, ox] = int(Tile.MOVING_OBS)
        ax, ay = self.agent_pos
        grid[ay, ax] = int(Tile.AGENT)
        lines = ["".join(glyphs[Tile(int(t))] for t in row) for row in grid]
        out = "\n".join(lines)
        return out

    def _render_rgb(self) -> np.ndarray:
        grid = self.grid.copy()
        for (ox, oy) in self.dynamic_obs:  # overlay dynamic obstacles first
            if 0 <= ox < self.config.width and 0 <= oy < self.config.height:
                grid[oy, ox] = int(Tile.MOVING_OBS)
        ax, ay = self.agent_pos
        grid[ay, ax] = int(Tile.AGENT)
        rgb = PALETTE[np.clip(grid, 0, len(PALETTE)-1)]
        # upscale each cell to 8x8 pixels
        rgb = np.kron(rgb, np.ones((8,8,1), dtype=np.uint8))
        return rgb


# ----------------------------
# Convenience factory
# ----------------------------

def make_env(**kwargs) -> AdvancedGridworldEnv:
    cfg = GridworldConfig(**kwargs)
    return AdvancedGridworldEnv(cfg)


# ----------------------------
# Simple sanity test (manual)
# ----------------------------

if __name__ == "__main__":
    env = make_env(width=12, height=12, view_size=5,
                   apples=1, axes=1, buckets=1, doors=1, dynamic_obstacles=1, keys=1,
                   lava_pools=1, picks=1, trees=2, water_pools=2,
                   walls_density=0.2, slip_prob=0.05, wind_strength=0.1, wind_dir=(1,0),
                   observation_mode="dict", render_mode="ansi")
    #obs, info = env.reset(seed=0)
    obs, info = env.reset()
    done = False
    total = 0.0
    while True:
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        total += r
        print(env.render())
        if term or trunc:
            print(f"Episode done, terminated: {term}, truncated: {trunc}, total reward: {total}")
            break
