from abc import abstractmethod

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from gym_sailing.utils.renderer import Renderer


class BoatEnv(gym.Env):
    BOAT_BEAM = 1.4  # meters
    BOAT_LENGTH = 4.2  # meters
    TARGET_RAD = BOAT_LENGTH / 2
    COURSE_SIZE = 50  # meters
    N_BOATS = 80
    TARGET = (COURSE_SIZE * 0.5, COURSE_SIZE * 0.90)

    metadata = {"render_modes": ["human", "rgb_array", "ansi"], "render_fps": 60}

    def __init__(self, render_mode=None):
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.renderer = None

        self.low = np.array(
            [
                -10,
                -np.pi,
                -1,
                -np.pi,
                0,
            ]
        )

        self.high = np.array(
            [
                10,
                np.pi,
                1,
                np.pi,
                self.COURSE_SIZE * 2,
            ]
        )

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float64)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float64)

    def step(self, action):
        self.stepnum += 1
        action = np.clip(action, -1, 1)
        self.last_action = action[0]
        self.boat.command(action[0])

        obs, distance2target = self._get_obs()
        terminated, reward = self._get_reward(distance2target)

        if self.render_mode == "human":
            self._render_frame()

        return (
            obs,
            reward,
            terminated,
            False,
            {
                "distance2target": np.linalg.norm(distance2target),
            },
        )

    def _get_reward(self, distance2target):
        terminated = False
        reward = -0.1  # Alive penalty

        if np.linalg.norm(distance2target) < self.TARGET_RAD:
            reward = 100
            terminated = True

        if np.linalg.norm(distance2target) >= self.COURSE_SIZE:
            reward = -100
            terminated = True

        # infinity norm previus distance to target- current distance to target
        reward += 10 * (
            np.linalg.norm(self.prev_distance2target, 8)
            - np.linalg.norm(distance2target, 8)
        )

        self.prev_distance2target = distance2target
        self.last_reward = reward
        return terminated, reward

    def _get_obs(self):
        distance2target = np.array([self.boat.x, self.boat.y]) - np.array(self.TARGET)
        heading2target = np.arctan2(
            self.TARGET[1] - self.boat.y, self.TARGET[0] - self.boat.x
        )

        obs = np.array(
            [
                self.boat.speed,
                norm(self.boat.heading - np.pi / 2),
                self.boat.heading_dot,
                norm(heading2target - np.pi / 2),
                2 * np.linalg.norm(distance2target) / self.COURSE_SIZE,
            ]
        )

        return obs, distance2target

    @abstractmethod
    def reset(self, options=None, seed=None):
        super().reset(seed=seed)
        self.stepnum = 0
        self.last_reward = 0
        self.last_action = 0
        if self.render_mode in ["human", "rgb_array"]:
            self.renderer = Renderer(
                self.BOAT_LENGTH, self.BOAT_BEAM, self.TARGET_RAD, self.COURSE_SIZE
            )

        self.prev_distance2target = np.array([self.boat.x, self.boat.y]) - np.array(
            self.TARGET
        )

        obs, _ = self._get_obs()

        return (obs, {})

    def render(self):
        if self.render_mode == "rgb_array" and self.renderer is not None:
            return self._render_frame()

        elif self.render_mode == "ansi":
            print(
                f"Speed: {self.boat.speed:.2f}, Heading: {self.boat.heading:.2f}, Heading Dot: {self.boat.heading_dot:.2f}, Heading to Target: {self.boat.heading2target:.2f}, Distance to Target: {self.boat.distance2target:.2f}"
            )

    def _render_frame(self):
        return self.renderer._render_frame(
            boats=[
                (
                    self.boat.x,
                    self.boat.y,
                    self.boat.heading - np.pi / 2,
                    self.last_action,
                )
            ],
            target=self.TARGET,
            stepnum=self.stepnum,
            reward=self.last_reward,
            render_mode=self.render_mode,
            fps=self.metadata["render_fps"],
        )

    def close(self):
        if self.renderer is not None:
            self.renderer.close()


def norm(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


# Discrete action space version of BoatEnv
class BoatDiscreteEnv(BoatEnv):
    def __init__(self, render_mode=None):
        super().__init__(render_mode)
        self.action_space = spaces.Discrete(3)

    def step(self, action):
        action = action - 1
        return super().step([action])
        return super().step([action])
