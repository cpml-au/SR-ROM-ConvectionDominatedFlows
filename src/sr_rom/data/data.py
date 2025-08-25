import numpy as np
from typing import List
import os


def process_data(r: int, bench_name: str, Re_list: List | str, num_t: int):
    """Function that transform ROM data in numpy arrays.

    Args:
        r: number of modes.
        bench_name: name of the directory where the data are located
        Re_list: list of Reynolds subdirectories to be used to build the
            ROM numpy arrays.
        num_t: number of time instances.

    Returns:
        the Reynolds, tau and a_FOM numpy arrays (first three numpy arrays).
    """

    data_path = os.path.dirname(os.path.realpath(__file__))
    bench_path = os.path.join(data_path, bench_name)

    dir_list = sorted(os.listdir(bench_path))
    Re, tau, a_FOM = [], [], []

    for directory in dir_list:
        directory_path = os.path.join(bench_path, directory)
        curr_Re = float(directory.replace("Re", ""))
        uk = np.loadtxt(directory_path + "/uk", delimiter=",")
        num_basis_functions = int(len(uk) / num_t)
        curr_a_FOM = uk.reshape((num_t, num_basis_functions))[:, 1 : (r + 1)]

        if curr_Re in Re_list:
            Re.append(curr_Re)
            a_FOM.append(curr_a_FOM)
            curr_tau = np.loadtxt(
                directory_path + "/vmsrom_clousre_N" + str(r) + "_all.txt",
                delimiter=",",
                usecols=range(r),
            )
            tau.append(curr_tau)

    Re = np.array(Re)
    tau = np.array(tau)
    a_FOM = np.array(a_FOM)

    return Re, tau, a_FOM
