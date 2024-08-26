import numpy as np
from gym_sailing.physics.boat import Boat, perpendicular, unit_vector


class MotorBoat(Boat):
    def __init__(self, x, y, heading, heading_dot=0.0, speed=0.0):
        super().__init__(x, y, heading, heading_dot, speed)

    def command(self, rudder):
        unit_heading = unit_vector(self.heading)

        speed = np.dot(
            self.velocity, unit_heading
        )  # positive means forward, negative means backward

        if speed > 0:
            sqrtspeed = np.sqrt(np.linalg.norm(self.velocity))
        else:
            sqrtspeed = -np.sqrt(np.linalg.norm(self.velocity))

        self.heading_dot *= 0.97
        if -self.MAX_ANGULAR_VELOCITY < self.heading_dot < self.MAX_ANGULAR_VELOCITY:
            self.heading_dot += -rudder * self.RUDDER_COEFF * sqrtspeed

        self.heading += self.heading_dot
        fcentripetal = self.heading_dot * self.mass

        unit_heading = unit_vector(self.heading)  # new heading
        unit_perp = perpendicular(unit_heading)

        fdrive = 100 * unit_vector(self.heading)

        vforward = np.dot(self.velocity, unit_heading) * unit_heading
        vperpendicular = self.velocity - vforward

        fdrag = (
            -vforward * np.linalg.norm(vforward) * 100.0
        )  # opposite to direction of movement
        fkeel = -vperpendicular * np.linalg.norm(vperpendicular) * 1200.0
        fperp = unit_perp * fcentripetal * np.linalg.norm(self.velocity)

        self.velocity += (fdrive + fdrag + fkeel + fperp) / self.mass

        delta_x = self.velocity[0]
        delta_y = self.velocity[1]
        self.x = self._update_state(self.x, delta_x)
        self.y = self._update_state(self.y, delta_y)
        self.speed = np.linalg.norm(self.velocity)

        return self.x, self.y, self.heading - np.pi / 2
