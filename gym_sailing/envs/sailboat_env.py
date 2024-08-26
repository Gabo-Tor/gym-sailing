import numpy as np

from gym_sailing.envs.boat_env import BoatDiscreteEnv, BoatEnv
from gym_sailing.physics.sailboat import SailBoat


class SailboatEnv(BoatEnv):
    def __init__(self, render_mode=None):
        super().__init__(render_mode)

    def reset(self, options=None, seed=None):
        self.boat = SailBoat(
            x=self.COURSE_SIZE * (0.5 + np.random.uniform(-0.2, 0.2)),
            y=self.COURSE_SIZE * 0.10,
            heading=self.np_random.random() * np.pi * 2,
            heading_dot=np.random.uniform(-0.03, 0.03),
            speed=np.random.uniform(-1, 0.5),
        )
        return super().reset(options, seed)


class SailboatDiscreteEnv(BoatDiscreteEnv):
    def __init__(self, render_mode=None):
        super().__init__(render_mode)

    def reset(self, options=None, seed=None):
        self.boat = SailBoat(
            x=self.COURSE_SIZE * (0.5 + np.random.uniform(-0.2, 0.2)),
            y=self.COURSE_SIZE * 0.10,
            heading=self.np_random.random() * np.pi * 2,
            heading_dot=np.random.uniform(-0.03, 0.03),
            speed=np.random.uniform(-1, 0.5),
        )
        return super().reset(options, seed)
