import numpy as np
from gym_sailing.envs.boat_env import BoatEnv
from gym_sailing.physics.motorboat import MotorBoat


class MotorboatEnv(BoatEnv):
    def __init__(self, render_mode=None):
        super().__init__(render_mode)

    def reset(self, options=None, seed=None):
        self.boat = MotorBoat(
            x=self.COURSE_SIZE * (0.5 + np.random.uniform(-0.2, 0.2)),
            y=self.COURSE_SIZE * 0.10,
            heading=self.np_random.random() * np.pi * 2,
            heading_dot=np.random.uniform(-0.03, 0.03),
            speed=np.random.uniform(-1, 0.5),
        )
        return super().reset(options, seed)

    def _render_frame(self):
        return self.renderer._render_frame(
            boats=[
                (
                    self.boat.x,
                    self.boat.y,
                    self.boat.heading - np.pi / 2,
                    self.last_action,
                    "motorboat",
                )
            ],
            target=self.TARGET,
            stepnum=self.stepnum,
            reward=self.last_reward,
            render_mode=self.render_mode,
            fps=self.metadata["render_fps"],
        )
