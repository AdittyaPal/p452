#!/usr/bin/env python3

"""
Classical Spin System Monte Carlo simulator
"""

import numpy as np
from numba import jit

from skdylib.spherical_coordinates import xyz_sph_urand

USE_NUMBA = True


class SpinSystem:
    """
    This class represent a spin system with Heisenberg interaction, DMI and Zeeman terms
    """

    def __init__(self, initial_state, J, D, Hz, T):
        self.J = J
        self.D = D

        self.Hz = Hz
        self.T = T
        self.beta = 1 / T

        self.state = initial_state
        self.nx = self.state.shape[0]
        self.ny = self.state.shape[1]
        self.nz = self.state.shape[2]
        self.sites_number = self.nx * self.ny * self.nz

        # Set the best energy functions depending on the the presence of an antisymmetric term
        self.step_function = compute_step_Heisenberg
        self.compute_energy_function = compute_energy_Heisenberg

        # Compute energy and magnetization of the initial initial_state
        self.energy = self.compute_energy_function(self.state, self.nx, self.ny, self.nz, J, D, Hz)
        self.total_magnetization = compute_magnetization(self.state)

    @property
    def magnetization(self):
        """
        The magnetization of the system
        :return: The value of the magnetization
        """
        return self.total_magnetization / self.sites_number

    def step(self):
        """
        Evolve the system computing a step of Metropolis-Hastings Monte Carlo.
        It actually calls the non-object oriented procedure.
        """
        s, e, m = self.step_function(self.state, self.nx, self.ny, self.nz, self.J, self.D, self.Hz, self.beta, self.energy, self.total_magnetization)
        self.state = s
        self.energy = e
        self.total_magnetization = m


# Compiled functions

@jit(nopython=USE_NUMBA, cache=USE_NUMBA)
def compute_magnetization(state):
    """
    Compute the total magnetization
    :return: [Mx, My, Mz] vector of mean magnetization
    """
    mx = np.sum(state[:,:,:,0])
    my = np.sum(state[:,:,:,1])
    mz = np.sum(state[:,:,:,2])
    return np.array([mx,my,mz])


# Pure Heisenberg model

@jit(nopython=USE_NUMBA, cache=USE_NUMBA)
def neighbhour_energy_Heisenberg(i, j, k, ii, jj, kk, state, J):
    """
    Compute the energy of two adjacent spins due to the Heisenberg Hamiltonian
    :return: the energy computed
    """
    heisenberg_term = -J*np.dot(state[i, j, k], state[ii, jj, kk])
    return heisenberg_term

@jit(nopython=USE_NUMBA, cache=USE_NUMBA)
def compute_energy_Heisenberg(state, nx, ny, nz, J, D, Hz):
    """
    Compute the energy of the system with Heisenberg Hamiltonian
    :return: The value of the energy
    """

    energy_counter = 0.0

    for i, j, k in np.ndindex(nx, ny, nz):
        ii = (i + 1) % nx
        energy_counter += neighbhour_energy_Heisenberg(i, j, k, ii, j, k, state, J)

        jj = (j + 1) % ny
        energy_counter += neighbhour_energy_Heisenberg(i, j, k, i, jj, k, state, J)

        if nz > 1:
            kk = (k + 1) % nz
            energy_counter += neighbhour_energy_Heisenberg(i, j, k, i, j, kk, state, J)

        energy_counter += - Hz * state[i, j, k, 2]

    return energy_counter


@jit(nopython=USE_NUMBA, cache=USE_NUMBA)
def compute_step_Heisenberg(state, nx, ny, nz, J, D, Hz, beta, energy, total_magnetization):
    """
    Evolve the system computing a step of Metropolis-Hastings Monte Carlo.
    This non OOP function is accelerated trough jit compilation.
    """

    # Select a random spin in the system
    i = np.random.randint(0, nx)
    j = np.random.randint(0, ny)
    k = np.random.randint(0, nz)

    # Compute the energy due to that spin
    e0 = 0

    ii = (i + 1) % nx
    e0 += neighbhour_energy_Heisenberg(i, j, k, ii, j, k, state, J)

    ii = (i - 1) % nx
    e0 += neighbhour_energy_Heisenberg(i, j, k, ii, j, k, state, J)

    jj = (j + 1) % ny
    e0 += neighbhour_energy_Heisenberg(i, j, k, i, jj, k, state, J)

    jj = (j - 1) % ny
    e0 += neighbhour_energy_Heisenberg(i, j, k, i, jj, k, state, J)

    if nz > 1:
        kk = (k + 1) % nz
        e0 += neighbhour_energy_Heisenberg(i, j, k, i, j, kk, state, J)

        kk = (k - 1) % nz
        e0 += neighbhour_energy_Heisenberg(i, j, k, i, j, kk, state, J)

    e0 += -Hz * state[i, j, k, 2]

    # Generate a new random direction and compute energy due to the spin in the new direction
    old_spin = state[i, j, k].copy()
    state[i, j, k] = xyz_sph_urand()

    e1 = 0

    ii = (i + 1) % nx
    e1 += neighbhour_energy_Heisenberg(i, j, k, ii, j, k, state, J)

    ii = (i - 1) % nx
    e1 += neighbhour_energy_Heisenberg(i, j, k, ii, j, k, state, J)

    jj = (j + 1) % ny
    e1 += neighbhour_energy_Heisenberg(i, j, k, i, jj, k, state, J)

    jj = (j - 1) % ny
    e1 += neighbhour_energy_Heisenberg(i, j, k, i, jj, k, state, J)

    if nz > 1:
        kk = (k + 1) % nz
        e1 += neighbhour_energy_Heisenberg(i, j, k, i, j, kk, state, J)

        kk = (k - 1) % nz
        e1 += neighbhour_energy_Heisenberg(i, j, k, i, j, kk, state, J)

    e1 += -Hz * state[i, j, k, 2]


    # Apply Metropolis algorithm
    w = np.exp(beta * (e0 - e1))
    r = np.random.uniform(0, 1)

    if r < w:
        energy += (e1 - e0)
        total_magnetization += state[i, j, k] - old_spin

    else:
        state[i,j,k] = old_spin

    return state, energy, total_magnetization
