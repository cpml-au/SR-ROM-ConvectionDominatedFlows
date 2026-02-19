from sr_rom.data.data import process_data
import numpy as np
import os
from sr_rom.fit_models import *
import warnings
import joblib
import torch
import os
import numpy.typing as npt
from sr_rom.roms import sr_rom
import argparse

# suppress warnings
warnings.filterwarnings("ignore")


def save_sr_results(
    X: npt.NDArray,
    y: npt.NDArray,
    sr_rom: sr_rom,
    Re: float,
    sdir: str,
    sr_directory: str,
    dt: float,
    iostep: int,
    T_sample_full: int,
    r: int,
    num_run: int,
):
    # save best models
    exist_models = os.path.exists(f"{sr_directory}models.txt")
    if exist_models:
        os.remove(f"{sr_directory}models.txt")

    tau_pred = sr_rom.predict(X)
    ucoefs_sr, energy_sr = solve_ode(
        Re, "SR", sdir, sr_directory, T_sample_full, dt, iostep, r, sr_rom
    )
    energy_MSE = np.mean(
        (energy_sr[val_idx] - energy_FOM[val_idx]) ** 2 / (energy_FOM[val_idx]) ** 2
    )

    # save results
    with open(f"{sr_directory}models_{str(num_run)}.txt", "a") as text_file:
        for i in range(r):
            str_model = sr_rom.reg[i].get_model_string(
                sr_rom.reg[i].model_, precision=8
            )
            text_file.write("tau_" + str(i) + " = " + str_model + "\n")
    np.save(f"{sr_directory}tau_pred_sr_{str(num_run)}.npy", tau_pred)
    np.save(f"{sr_directory}energy_sr_{str(num_run)}.npy", energy_sr)
    np.save(f"{sr_directory}ucoefs_sr_{str(num_run)}.npy", ucoefs_sr)
    return energy_MSE


# lr fit
def compute_lr_results(
    X: npt.NDArray,
    y: npt.NDArray,
    val_idx: npt.NDArray,
    sdir: str,
    lr_directory: str,
    r: int,
    Re: float,
    T_sample: int,
    dt: float,
    iostep: int,
    T_sample_full,
    energy_FOM: npt.NDArray,
):
    lr_rom = fit_lr(
        X, y, val_idx, sdir, lr_directory, r, Re, T_sample, dt, iostep, energy_FOM
    )
    # save best models
    tau_pred = lr_rom.predict(X)
    for i in range(r):
        joblib.dump(lr_rom.reg[i], lr_directory + "model_" + str(i) + ".pkl")
    ucoefs_lr, energy_lr = solve_ode(
        Re, "LR", sdir, lr_directory, T_sample_full, dt, iostep, r
    )
    np.save(f"{lr_directory}energy_lr.npy", energy_lr)
    np.save(f"{lr_directory}tau_pred_lr.npy", tau_pred)
    np.save(f"{lr_directory}ucoefs_lr.npy", ucoefs_lr)
    print(np.mean((energy_lr[val_idx] - energy_FOM[val_idx]) ** 2))
    print("LR-ROM results computed!")


# sr fit
def compute_sr_results(
    X: npt.NDArray,
    y: npt.NDArray,
    val_idx: npt.NDArray,
    sdir: str,
    sr_directory: str,
    r: int,
    Re: float,
    T_sample: int,
    dt: float,
    iostep: int,
    T_sample_full: int,
    energy_FOM: npt.NDArray,
    num_runs: int,
):
    print("SR-ROM training started", flush=True)
    for num_run in np.arange(num_runs):
        sr_rom = fit_sr(
            X, y, val_idx, sdir, sr_directory, r, Re, T_sample, dt, iostep, energy_FOM
        )

        energy_MSE = save_sr_results(
            X, y, sr_rom, Re, sdir, sr_directory, dt, iostep, T_sample_full, r, num_run
        )
        print(f"Best relative MSE in run {num_run}: {energy_MSE}")

    print("SR-ROM results computed!", flush=True)


def compute_nn_results(
    X: npt.NDArray,
    y: npt.NDArray,
    val_idx: npt.NDArray,
    sdir: str,
    nn_directory: str,
    r: int,
    Re: float,
    T_sample,
    dt: float,
    iostep: int,
    T_sample_full,
    energy_FOM: npt.NDArray,
    device: str,
    num_runs: int,
):
    print("NN-ROM training started", flush=True)
    for num_run in range(num_runs):
        nn_rom, nn_best_dict = fit_nn(
            X,
            y,
            val_idx,
            sdir,
            nn_directory,
            r,
            Re,
            T_sample,
            dt,
            iostep,
            energy_FOM,
            device,
        )

        # save best models
        with open(f"{nn_directory}params.txt", "a") as text_file:
            for i in range(r):
                nn_rom.reg[i].save_params(f"{nn_directory}model_param_{str(i)}.pkl")
                print(nn_best_dict, file=text_file)

        # X_nn = torch.from_numpy(X).to(torch.float32)
        tau_pred = nn_rom.predict(X)

        ucoefs_nn, energy_nn = solve_ode(
            Re, "NN", sdir, nn_directory, T_sample_full, dt, iostep, r, nn_rom
        )
        print(np.mean((energy_nn[val_idx] - energy_FOM[val_idx]) ** 2), flush=True)

        # save results
        os.rename(
            f"{nn_directory}params.txt", f"{nn_directory}params_{str(num_run)}.txt"
        )
        for i in range(r):
            os.rename(
                f"{nn_directory}model_param_{str(i)}.pkl",
                f"{nn_directory}model_param_{str(i)}_{str(num_run)}.pkl",
            )
        np.save(f"{nn_directory}tau_pred_nn_{str(num_run)}.npy", tau_pred)
        np.save(f"{nn_directory}energy_nn_{str(num_run)}.npy", energy_nn)
        np.save(f"{nn_directory}ucoefs_nn_{str(num_run)}.npy", ucoefs_nn)

    print("NN-ROM results computed!", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench_name", type=str, required=True)
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--re", type=str, required=True)

    args = parser.parse_args()
    bench_name = f"{args.bench_name}/"
    method = args.method
    fixed_Re = int(args.re)

    if bench_name == "2dcyl/":
        # load all the data
        Re_list = [400, 500]
        # parameter for a given Reynolds in Re_list
        T_sample = 40
        T_sample_full = 100
        dt = 0.01
        iostep = 1
    elif bench_name == "ldc/":
        Re_list = [10000, 15000, 20000]
        # parameter for a given Reynolds in Re_list
        T_sample = 160
        T_sample_full = 400
        dt = 0.01
        iostep = 4

    val_idx = np.arange(2001, 4001)
    num_t = 2001
    r = 2
    Re, tau, a_FOM = process_data(r, f"{bench_name}Re", Re_list, num_t)

    idx_Re = np.where(Re == fixed_Re)[0][0]
    X = a_FOM[idx_Re, :]
    y = tau[idx_Re, :]

    num_runs = 5

    # Standardization
    mean_std_X_train = [np.mean(X, axis=0), np.std(X, axis=0)]
    X = (X - mean_std_X_train[0]) / mean_std_X_train[1]

    main_path = os.path.dirname(os.path.realpath(__file__))
    # path to the NekROM operators
    sdir = os.path.join(main_path, f"data/{bench_name}Re/Re{str(fixed_Re)}")
    # path to FOM
    fom_dir = os.path.join(
        main_path, f"data/{bench_name}fom_energies/Re{str(fixed_Re)}"
    )

    fom_length = sum(1 for _ in open(f"{fom_dir}/fom_energy"))
    energy_FOM = np.zeros(fom_length)
    with open(f"{fom_dir}/fom_energy", "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        energy_FOM[i] = line.split()[-2]

    # define directories in which the results are saved
    results_dir = main_path + bench_name + "r_" + str(r) + "/"
    if method == "LR":
        lr_directory = f"{results_dir}lr/Re{str(fixed_Re)}/"
        os.makedirs(lr_directory, exist_ok=True)
        np.save(f"{lr_directory}mean_std_X_train.npy", mean_std_X_train)
        compute_lr_results(
            X,
            y,
            val_idx,
            sdir,
            lr_directory,
            r,
            fixed_Re,
            T_sample,
            dt,
            iostep,
            T_sample_full,
            energy_FOM,
        )
    elif method == "SR":
        sr_directory = f"{results_dir}sr/Re{str(fixed_Re)}/"
        os.makedirs(sr_directory, exist_ok=True)
        np.save(f"{sr_directory}mean_std_X_train.npy", mean_std_X_train)
        compute_sr_results(
            X,
            y,
            val_idx,
            sdir,
            sr_directory,
            r,
            fixed_Re,
            T_sample,
            dt,
            iostep,
            T_sample_full,
            energy_FOM,
            num_runs,
        )
    elif method == "NN":
        nn_directory = f"{results_dir}nn/Re{str(fixed_Re)}/"
        os.makedirs(nn_directory, exist_ok=True)
        np.save(f"{nn_directory}mean_std_X_train.npy", mean_std_X_train)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        compute_nn_results(
            X,
            y,
            val_idx,
            sdir,
            nn_directory,
            r,
            fixed_Re,
            T_sample,
            dt,
            iostep,
            T_sample_full,
            energy_FOM,
            device,
            num_runs,
        )
