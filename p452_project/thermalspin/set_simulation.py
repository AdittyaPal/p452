"""
Run multiple instance of Heisenberg
"""

import getopt
import os
import sys
import time
from multiprocessing import pool, cpu_count

import numpy as np

from skdylib.counter import Counter
from thermalspin.read_config import read_config_file
from thermalspin.simulation import init_simulation_aligned, init_simulation_random, run_simulation

DEFAULT_PARAMS, SIMULATIONS_DIRECTORY, PROCESSES_NUMBER = None, None, None


def init_set(setname, J, Hz, T, L, S_0=None):
    simdir_list = []
    params_list = []
    L_list = []
    set_directory = SIMULATIONS_DIRECTORY + setname + "/"

    for i, j, k in np.ndindex((T.shape[0], L.shape[0], Hz.shape[0])):
        T_str = "{0:.3f}".format(T[i])
        Hz_str = "{0:.3f}".format(Hz[k])
        simdir_list.append(set_directory + setname + f"_T{T_str}_L{L[j]}_H{Hz_str}/")
        params = DEFAULT_PARAMS
        params["param_T"] = [float(T[i])]
        params["param_Hz"] = [float(Hz[k])]
        params["param_J"] = [float(J[0])]
        params_list.append(params.copy())
        L_list.append(L[j].copy())

    if S_0 is None:
        for i in range(len(simdir_list)):
            init_simulation_random(simdir_list[i], nx=L_list[i], ny=L_list[i], nz=L_list[i], params=params_list[i])

    else:
        for i in range(len(simdir_list)):
            init_simulation_aligned(simdir_list[i], nx=L_list[i], ny=L_list[i], nz=L_list[i], params=params_list[i], S_0=S_0)


def init_2D_set(setname, J, Hz, T, L, S_0=None):
    simdir_list = []
    params_list = []
    L_list = []
    set_directory = SIMULATIONS_DIRECTORY + setname + "/"

    for i, j, k in np.ndindex((T.shape[0], L.shape[0], Hz.shape[0])):
        T_str = "{0:.3f}".format(T[i])
        Hz_str = "{0:.3f}".format(Hz[k])
        simdir_list.append(set_directory + setname + f"_T{T_str}_L{L[j]}_H{Hz_str}/")
        params = DEFAULT_PARAMS
        params["param_T"] = [float(T[i])]
        params["param_Hz"] = [float(Hz[k])]
        params["param_J"] = [float(J[0])]
        params_list.append(params.copy())
        L_list.append(L[j].copy())

    if S_0 is None:
        for i in range(len(simdir_list)):
            init_simulation_random(simdir_list[i], nx=L_list[i], ny=L_list[i], nz=1, params=params_list[i])

    else:
        for i in range(len(simdir_list)):
            init_simulation_aligned(simdir_list[i], nx=L_list[i], ny=L_list[i], nz=1, params=params_list[i], S_0=S_0)



# Run

simulations_number = 0
completed_simulations_counter = Counter()


def run_simulation_wrapper(simdir):
    run_simulation(simdir, verbose=False)

    completed_simulations_counter.increment()
    completed_simulations_number = completed_simulations_counter.value()

    print(f"Completed simulations {completed_simulations_number}/{simulations_number}")


def run_set(set_name):
    path = SIMULATIONS_DIRECTORY + set_name + "/"
    file_list = [f for f in os.listdir(path) if not f.startswith('.')]
    file_list.sort()

    simdir_list = []

    for filename in file_list:
        simdir_list.append(SIMULATIONS_DIRECTORY + set_name + "/" + filename + "/")

    global simulations_number
    simulations_number = len(simdir_list)

    if PROCESSES_NUMBER <= 0:
        processes_number = cpu_count() + PROCESSES_NUMBER
    else:
        processes_number = PROCESSES_NUMBER

    processes_pool = pool.Pool(processes=processes_number)
    processes_pool.map(run_simulation_wrapper, simdir_list)


def set_simulation():
    global DEFAULT_PARAMS, SIMULATIONS_DIRECTORY, PROCESSES_NUMBER
    DEFAULT_PARAMS, SIMULATIONS_DIRECTORY, PROCESSES_NUMBER = read_config_file()

    mode = None
    setname = None
    L = []
    J = DEFAULT_PARAMS["param_J"]
    T = np.array(DEFAULT_PARAMS["param_T"])
    Hz = np.array(DEFAULT_PARAMS["param_Hz"])
    sim_2D = False
    theta_0, phi_0 = (None, None)
    Ti, Tf, dt = (None, None, None)

    try:
        opts, args = getopt.getopt(sys.argv[1:], "hr:i:L:m:T:J:H:",
                                   ["help", "initialize=", "run=", "2D", "temperatures="])
    except getopt.GetoptError:
        #usage()
        sys.exit(2)

    for opt, arg in opts:
        '''
        if opt in ("-h", "--help"):
            usage()
            sys.exit()
        '''
        if opt in ("-r", "--run"):
            mode = "run"
            setname = arg
        elif opt in ("-i", "--initialize"):
            mode = "init"
            setname = arg

        elif opt in ("-L"):
            for dim in arg.split(","):
                L.append(int(dim))
            L = np.array(L)

        elif opt in ("-m", "--magnetization"):
            theta_0, phi_0 = arg.split(",")

        elif opt in ("-T", "--temperature"):
            if len(arg.split(",")) == 3:
                Ti, Tf, dT = arg.split(",")
                Ti = float(Ti)
                Tf = float(Tf)
                dT = float(dT)
                T = np.arange(Ti, Tf, dT)
            elif len(arg.split(",")) == 1:
                T = np.array([float(arg)])

        elif opt in ("-H"):
            if len(arg.split(",")) == 3:
                Hi, Hf, dH = arg.split(",")
                Hi = float(Hi)
                Hf = float(Hf)
                dH = float(dH)
                Hz = np.arange(Hi, Hf, dH)
            elif len(arg.split(",")) == 1:
                Hz = np.array([float(arg)])

        elif opt in "-J":
            J = np.array([float(arg)])

        elif opt in "--2D":
            sim_2D = True

    if mode == "run":
        print(f"Running simulations in set {setname}")
        starting_time = time.time()
        run_set(setname)
        total_time = time.time() - starting_time
        total_time_str = "{0:.2f}".format(total_time)
        print(f"Total running time: {total_time_str} s")

    elif mode == "init":
        if sim_2D:
            init_2D_set(setname, J, Hz, T, L)
        elif theta_0 is None:
            init_set(setname, J, Hz, T, L)
        else:
            init_set(setname, J, Hz, T, L, theta_0, phi_0)
    else:
        sys.exit(2)

    print("Finished")
