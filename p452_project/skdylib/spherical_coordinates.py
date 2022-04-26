"""
Mathematical utility functions
"""

import numpy as np
from numba import jit


@jit(nopython=True, cache=True)
def xyz2sph(v):
    """
    Convert a unit vector from cartesian coordinates to spherical coordinates
    :param v: unit vector
    :return: [theta, phi] polar coordinates
    """

    return np.array([np.arccos(v[2]), np.arctan2(v[1], v[0])])

@jit(nopython=True, cache=True)
def xyz_sph_urand():
    """
    Generate random unit vector in cartesian coordinates
    :return: [x, y, z] the unit vector
    """

    phi = np.random.uniform(0, 2 * np.pi)
    u = np.random.uniform(0, 1)
    theta = np.arccos(2 * u - 1)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.array([x, y, z])

