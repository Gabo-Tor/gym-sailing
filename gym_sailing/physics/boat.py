"""Physics of a generic boat"""

from abc import ABC, abstractmethod

import numpy as np


def norm(angle: float) -> float:
    """Normalize angle to be between -pi and pi"""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def unit_vector(angle):
    """Returns the unit vector for a given angle in radians."""
    return np.array([np.cos(angle), np.sin(angle)])


def perpendicular(a):
    """Returns a perpendicular vector to the given 2D vector a."""
    return np.array([-a[1], a[0]])


class Boat(ABC):
    TIME_STEP = 0.1  # seconds
    MAX_ANGULAR_VELOCITY = 300.0 / 360.0 * 2 * np.pi * TIME_STEP  # radians per second
    RUDDER_COEFF = 0.002

    def __init__(self, x, y, heading, heading_dot=0.0, speed=0.0):
        self.reset(x, y, heading, heading_dot, speed)
        self.velocity = np.array([0.0, 0.0])
        self.heading_dot = 0.0
        self.mass = 3000.0  # kg
        self.speed = 0.0

    @abstractmethod
    def command(self, rudder):
        pass

    def _update_state(self, value, delta_value):
        value += Boat.TIME_STEP * delta_value
        return value

    def reset(self, x, y, heading, heading_dot=0.0, speed=0.0):
        self.x = x
        self.y = y
        self.heading = heading
        self.heading_dot = heading_dot
        self.speed = speed

        self.speed = speed
