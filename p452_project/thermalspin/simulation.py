"""
Classical Heisenberg model Monte Carlo simulator
"""

import json
import os
import shutil
import time

import numpy as np

from skdylib.spherical_coordinates import xyz_sph_urand
from thermalspin.spin_system import SpinSystem

SNAPSHOTS_ARRAY_INITIAL_DIMENSION = int(3e4)


class TSSimulation:
    """
    Handler of the SpinSystem simulation.
    It runs the simulation and collects the results.
    """

    def __init__(self, hsys: SpinSystem, take_states_snapshots=False):
        """
        :param hsys: system to be evolved
        """

        self.system = hsys
        self.steps_counter = 0
        self.snapshots_counter = 0
        self.snapshots_array_dimension = SNAPSHOTS_ARRAY_INITIAL_DIMENSION
        self.snapshots = None

        self.snapshots_t = np.zeros(shape=SNAPSHOTS_ARRAY_INITIAL_DIMENSION)
        self.snapshots_e = np.zeros(shape=SNAPSHOTS_ARRAY_INITIAL_DIMENSION)
        self.snapshots_m = np.zeros(shape=(SNAPSHOTS_ARRAY_INITIAL_DIMENSION, 3))

        self.snapshots_J = np.zeros(shape=SNAPSHOTS_ARRAY_INITIAL_DIMENSION)
        self.snapshots_D = np.zeros(shape=SNAPSHOTS_ARRAY_INITIAL_DIMENSION)
        self.snapshots_T = np.zeros(shape=SNAPSHOTS_ARRAY_INITIAL_DIMENSION)
        self.snapshots_Hz = np.zeros(shape=SNAPSHOTS_ARRAY_INITIAL_DIMENSION)

        self.take_snapshot()

    def run(self, nsteps):
        """
        Evolve the system for a given number of steps
        :param nsteps: The number of steps
        """
        self.steps_counter += nsteps
        for t in range(1, nsteps + 1):
            self.system.step()

    def take_snapshot(self):
        """"
        Take a snapshot of the system, parameters and results
        """

        # First check if the snapshots array needs reshape
        if self.snapshots_counter == self.snapshots_array_dimension:
            self.snapshots = None

            self.snapshots_t.resize(self.snapshots_counter + SNAPSHOTS_ARRAY_INITIAL_DIMENSION)
            self.snapshots_e.resize(self.snapshots_counter + SNAPSHOTS_ARRAY_INITIAL_DIMENSION)
            self.snapshots_m.resize((self.snapshots_counter + SNAPSHOTS_ARRAY_INITIAL_DIMENSION, 3))

            self.snapshots_J.resize(self.snapshots_counter + SNAPSHOTS_ARRAY_INITIAL_DIMENSION)
            self.snapshots_D.resize(self.snapshots_counter + SNAPSHOTS_ARRAY_INITIAL_DIMENSION)
            self.snapshots_T.resize(self.snapshots_counter + SNAPSHOTS_ARRAY_INITIAL_DIMENSION)
            self.snapshots_Hz.resize(self.snapshots_counter + SNAPSHOTS_ARRAY_INITIAL_DIMENSION)

        self.snapshots_t[self.snapshots_counter] = self.steps_counter
        self.snapshots_e[self.snapshots_counter] = self.system.energy
        self.snapshots_m[self.snapshots_counter, :] = self.system.magnetization

        self.snapshots_J[self.snapshots_counter] = self.system.J
        self.snapshots_T[self.snapshots_counter] = self.system.T
        self.snapshots_D[self.snapshots_counter] = self.system.D
        self.snapshots_Hz[self.snapshots_counter] = self.system.Hz

        self.snapshots_counter += 1

    def run_with_snapshots(self, steps_number, delta_snapshots, verbose=False):
        """
        Evolve the system while taking snapshots
        :param steps_number: Number of steps to be computed
        :param delta_snapshots: Distance between snapshots
        """

        if steps_number % delta_snapshots != 0:
            raise Exception("steps_number must be multiple of delta_snapshots")

        nsnapshots = int(steps_number / delta_snapshots)
        for t in range(0, nsnapshots):
            self.run(delta_snapshots)
            self.take_snapshot()
            if verbose:
                print(f"Step number {self.steps_counter}", end="\r")


# Functions for initialization and saving to disk the results of a simulation

def init_simulation_aligned(simdir, nx, ny, nz, params, S_initial):
    """
    Generate a lattice of spins aligned toward an axis
    :param simdir: Directory of the simulation
    :param nx: Number of x cells
    :param ny: Number of y cells
    :param nz: Number of z cells
    :param params: parameters of the simulation
    :param phi_0:
    :param theta_0:
    """
    shutil.rmtree(simdir, ignore_errors=True)

    state = np.ones(shape=(nx, ny, nz, 3))
    state[:, :, :, 0] = S_initial[0]
    state[:, :, :, 1] = S_initial[1]
    state[:, :, :, 2] = S_initial[2]

    os.makedirs(simdir)
    params_file = open(simdir + "params.json", "w")
    json.dump(params, params_file, indent=2)
    np.save(simdir + "state.npy", state)


def init_simulation_random(simdir, nx, ny, nz, params):
    """
    Generate a lattice of spins aligned toward tan axis if specified, random if not
    :param simdir: Directory of the simulation
    :param nx: Number of x cells
    :param ny: Number of y cells
    :param nz: Number of z cells
    :param params: parameters of the simulation
    """
    shutil.rmtree(simdir, ignore_errors=True)

    state = np.zeros(shape=(nx, ny, nz, 3))
    for i, j, k in np.ndindex(nx, ny, nz):
        state[i, j, k] = xyz_sph_urand()

    os.makedirs(simdir)
    params_file = open(simdir + "params.json", "w")
    json.dump(params, params_file, indent=2)
    np.save(simdir + "state.npy", state)


def run_simulation(simulation_directory, verbose=True):
    """
    Run a simulation and save to disk the results
    :param simulation_directory: the directory of the simulation
    :param verbose: print step numbers in real time
    """

    if os.path.isfile(simulation_directory + "params.json"):
        params_file = open(simulation_directory + "params.json", "r")
        params = json.load(params_file)
    else:
        raise Exception("Missing params.json file")

    if os.path.isfile(simulation_directory + "state.npy"):
        state = np.load(simulation_directory + "state.npy")
    else:
        raise Exception("Missing state.npy file")

    param_J = np.array(params["param_J"], dtype=np.float)
    param_D = np.array(params["param_D"], dtype=np.float)
    param_Hz = np.array(params["param_Hz"], dtype=np.float)
    param_T = np.array(params["param_T"], dtype=np.float)
    steps_number = params["steps_number"]
    delta_snapshots = params["delta_snapshots"]
    save_snapshots = params["save_snapshots"]

    sys = SpinSystem(state, param_J[0], param_D[0], param_Hz[0], param_T[0])
    hsim = TSSimulation(sys, take_states_snapshots=save_snapshots)

    for i in range(param_T.shape[0]):
        T_str = "{0:.3f}".format(param_T[i])
        Hz_str = "{0:.3f}".format(param_Hz[i])
        print(f"Simulation stage:   {i}\n"
              f"Temperature:        {T_str}\n"
              f"Hz:                 {Hz_str}\n"
              f"Steps number:       {steps_number}\n"
              f"Delta snapshots:    {delta_snapshots}\n")

        hsim.system.J = param_J[i]
        hsim.system.D = param_D[i]
        hsim.system.T = param_T[i]
        hsim.system.Hz = param_Hz[i]
        start_time = time.time()
        hsim.run_with_snapshots(steps_number, delta_snapshots, verbose=verbose)
        end_time = time.time()
        run_time = end_time - start_time

        run_time_str = "{0:.2f}".format(run_time)
        print(f"Stage completed in {run_time_str} seconds\n")

    print("Saving results ...", end="")
    start = time.time()

    # Save the last state
    np.save(simulation_directory + "state.npy", hsim.system.state)

    # Collect the results of the simulation
    new_results = np.zeros(shape=(hsim.snapshots_counter, 4))
    new_results[:, 0] = hsim.snapshots_e[:hsim.snapshots_counter]
    new_results[:, 1:4] = hsim.snapshots_m[:hsim.snapshots_counter]

    # Collect the snapshots and params
    new_snapshots_params = np.zeros(shape=(hsim.snapshots_counter, 5))
    new_snapshots_params[:, 0] = hsim.snapshots_t[:hsim.snapshots_counter]
    new_snapshots_params[:, 1] = hsim.snapshots_J[:hsim.snapshots_counter]
    new_snapshots_params[:, 2] = hsim.snapshots_D[:hsim.snapshots_counter]
    new_snapshots_params[:, 3] = hsim.snapshots_Hz[:hsim.snapshots_counter]
    new_snapshots_params[:, 4] = hsim.snapshots_T[:hsim.snapshots_counter]

    # If old data is found, append the new one
    if os.path.isfile(simulation_directory + "snapshots_params.npy") and os.path.isfile(
            simulation_directory + "results.npy"):

        old_results = np.load(simulation_directory + "results.npy")
        results = np.concatenate((old_results, new_results[1:]))

        old_snapshots_params = np.load(simulation_directory + "snapshots_params.npy")
        last_t = old_snapshots_params[-1, 0]
        new_snapshots_params[:, 0] += last_t
        snapshots_params = np.concatenate((old_snapshots_params, new_snapshots_params[1:]))

    else:
        snapshots_params = new_snapshots_params
        results = new_results

    # Save all
    np.save(simulation_directory + "snapshots_params.npy", snapshots_params)
    np.save(simulation_directory + "results.npy", results)

    end = time.time()
    saving_time = end - start
    saving_time_str = "{0:.6f}".format(saving_time)

    print(f"done in {saving_time_str} seconds.")
