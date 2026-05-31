import random
import math
from typing import Dict, Optional

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame


class GridWorld(gym.Env):
    """NxM GridWorld environment.

    Observation: Dict with 'agent' and 'goal' positions as (row, col) ints.
    Action space: 0=up,1=right,2=down,3=left

    reward_type: 'sparse' (goal only), 'dense' (negative per-step + goal bonus),
                 'shaped' (distance-based shaping reward)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        N: int = 5,
        M: int = 5,
        reward_type: str = "sparse",
        max_steps: Optional[int] = None,
        render_mode: Optional[str] = None,
        visits: bool = False,
    ):
        assert N > 0 and M > 0, "Grid dimensions must be positive"
        assert reward_type in ("sparse", "dense", "shaped"), "invalid reward_type"
        assert render_mode in (None, "human", "rgb_array"), "invalid render_mode"
        assert isinstance(visits, bool), "visits must be a bool"

        super().__init__()
        self.N = N
        self.M = M
        self.reward_type = reward_type
        self.max_steps = max_steps or (N * M * 4)
        self.render_mode = render_mode
        self.visits = visits
        self.window_size = 1024
        self.window = None
        self.clock = None
        self.font = None

        # Spaces
        # agent and goal positions packed as two integers each
        coord_low = np.array([0, 0], dtype=np.int32)
        coord_high = np.array([N - 1, M - 1], dtype=np.int32)
        self.observation_space = spaces.Dict({
            "agent": spaces.Box(low=coord_low, high=coord_high, shape=(2,), dtype=np.int32),
            "goal": spaces.Box(low=coord_low, high=coord_high, shape=(2,), dtype=np.int32),
        })

        self.action_space = spaces.Discrete(4)

        # state
        self.agent_pos = (0, 0)
        self.goal_pos = (N - 1, M - 1)
        self.steps = 0
        self.episode_reward = 0.0
        self.path = [self.agent_pos]

    def seed(self, seed: Optional[int] = None):
        random.seed(seed)
        return [seed]

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.seed(seed)

        # randomize agent and goal but ensure they are not equal
        ax = random.randrange(self.N)
        ay = random.randrange(self.M)
        gx = random.randrange(self.N)
        gy = random.randrange(self.M)
        while gx == ax and gy == ay:
            gx = random.randrange(self.N)
            gy = random.randrange(self.M)

        self.agent_pos = (ax, ay)
        self.goal_pos = (gx, gy)
        self.steps = 0
        self.episode_reward = 0.0
        self.path = [self.agent_pos]
        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), {}

    def step(self, action: int):
        assert self.action_space.contains(action), "invalid action"
        prev_pos = self.agent_pos
        x, y = self.agent_pos
        if action == 0 and x > 0:
            x -= 1
        elif action == 1 and y < self.M - 1:
            y += 1
        elif action == 2 and x < self.N - 1:
            x += 1
        elif action == 3 and y > 0:
            y -= 1

        self.agent_pos = (x, y)
        if self.visits and self.agent_pos != prev_pos:
            self.path.append(self.agent_pos)
        self.steps += 1

        done = self.agent_pos == self.goal_pos or (self.steps >= self.max_steps)
        reward = self._compute_reward(done)
        self.episode_reward += reward
        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, done, False, {}

    def _compute_reward(self, done: bool) -> float:
        ax, ay = self.agent_pos
        gx, gy = self.goal_pos
        dist = abs(ax - gx) + abs(ay - gy)  # Manhattan distance

        if self.reward_type == "sparse":
            return 1.0 if self.agent_pos == self.goal_pos else 0.0

        if self.reward_type == "dense":
            # small negative per step, large positive on goal
            if self.agent_pos == self.goal_pos:
                return 10.0
            return -0.1

        # shaped: reward is delta in (negative) distance to goal
        # encourage moves that reduce manhattan distance
        # previous distance computed by rolling back one step is not stored, so
        # we approximate shaped reward as inverse distance scaled
        return (self.N + self.M) - dist

    def _get_obs(self) -> Dict[str, np.ndarray]:
        return {
            "agent": np.array(self.agent_pos, dtype=np.int32),
            "goal": np.array(self.goal_pos, dtype=np.int32),
        }

    def render(self):
        if self.render_mode is None:
            return None
        return self._render_frame()

    def _render_frame(self):
        grid_w = self.window_size // self.M
        grid_h = self.window_size // self.N
        cell_size = min(grid_w, grid_h)

        if self.render_mode == "human":
            if self.window is None:
                pygame.display.init()
                self.window = pygame.display.set_mode((self.window_size, self.window_size))
            if self.clock is None:
                self.clock = pygame.time.Clock()
            if self.font is None:
                pygame.font.init()
                self.font = pygame.font.Font(None, max(24, self.window_size // 32))

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((245, 245, 245))

        # Draw grid cells and lines.
        for row in range(self.N):
            for col in range(self.M):
                rect = pygame.Rect(col * cell_size, row * cell_size, cell_size, cell_size)
                pygame.draw.rect(canvas, (230, 230, 230), rect)
                pygame.draw.rect(canvas, (180, 180, 180), rect, width=1)

        ax, ay = self.agent_pos
        gx, gy = self.goal_pos

        if self.visits and len(self.path) > 1:
            path_color = (0, 180, 0)
            line_width = max(2, cell_size // 8)
            arrow_len = max(8, cell_size // 4)
            arrow_w = max(4, cell_size // 8)

            for i in range(len(self.path) - 1):
                start_rc = self.path[i]
                end_rc = self.path[i + 1]
                start = self._cell_center(start_rc, cell_size)
                end = self._cell_center(end_rc, cell_size)
                self._draw_arrow(canvas, start, end, path_color, line_width, arrow_len, arrow_w)

        agent_rect = pygame.Rect(ay * cell_size, ax * cell_size, cell_size, cell_size)
        goal_rect = pygame.Rect(gy * cell_size, gx * cell_size, cell_size, cell_size)

        pygame.draw.rect(canvas, (70, 170, 255), goal_rect)
        pygame.draw.circle(
            canvas,
            (220, 60, 60),
            agent_rect.center,
            max(4, cell_size // 3),
        )

        if self.render_mode == "human" and self.font is not None:
            hud_text = f"Reward: {self.episode_reward:.2f}  Steps: {self.steps}"
            label = self.font.render(hud_text, True, (20, 20, 20))
            label_rect = label.get_rect()
            label_rect.topleft = (12, 12)
            bg_rect = label_rect.inflate(12, 10)
            pygame.draw.rect(canvas, (255, 255, 255), bg_rect, border_radius=6)
            pygame.draw.rect(canvas, (90, 90, 90), bg_rect, width=1, border_radius=6)
            canvas.blit(label, label_rect)

        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return None

            self.window.blit(canvas, canvas.get_rect())
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
            return None

        return np.transpose(pygame.surfarray.array3d(canvas), axes=(1, 0, 2))

    def _cell_center(self, pos, cell_size: int):
        row, col = pos
        return (col * cell_size + (cell_size // 2), row * cell_size + (cell_size // 2))

    def _draw_arrow(
        self,
        surface,
        start,
        end,
        color,
        line_width: int,
        head_len: int,
        head_width: int,
    ):
        sx, sy = start
        ex, ey = end
        dx = ex - sx
        dy = ey - sy
        length = math.hypot(dx, dy)
        if length < 1e-6:
            return

        pygame.draw.line(surface, color, start, end, line_width)

        ux = dx / length
        uy = dy / length
        left = (
            ex - ux * head_len - uy * head_width,
            ey - uy * head_len + ux * head_width,
        )
        right = (
            ex - ux * head_len + uy * head_width,
            ey - uy * head_len - ux * head_width,
        )
        pygame.draw.polygon(surface, color, [(ex, ey), left, right])

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None
            self.font = None


def make(
    name: str,
    N: int,
    M: int,
    reward_type: str = "sparse",
    max_steps: Optional[int] = None,
    render_mode: Optional[str] = None,
    visits: bool = False,
):
    """Compatibility helper: gymnasium.make_env("GridWorld", N, M) -> env

    call: make_env("GridWorld", N, M, reward_type)
    """
    if name != "GridWorld":
        raise ValueError("This helper only knows how to make 'GridWorld'")
    return GridWorld(
        N=N,
        M=M,
        reward_type=reward_type,
        max_steps=max_steps,
        render_mode=render_mode,
        visits=visits,
    )


# If gymnasium has make_env, optionally patch it to accept our simple signature.
try:
    if not hasattr(gym, "make"):
        setattr(gym, "make", make)
except Exception:
    pass


__all__ = ["GridWorld", "make"]
