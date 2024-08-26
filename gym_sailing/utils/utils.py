import numpy as np


def unit_vector(angle):
    """Returns the unit vector for a given angle in radians."""
    return np.array([np.cos(angle), np.sin(angle)])


def angle_between(v1, v2):
    """Returns the angle in radians between vectors 'v1' and 'v2'."""
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def perpendicular(a):
    """Returns a perpendicular vector to the given 2D vector a."""
    return np.array([-a[1], a[0]])
