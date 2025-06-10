#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
After generating the data with command-line argument '--num_sol_per_npy' set,
the PDE solution data will be stored in a series of *.npy files, and the *.hdf5
file only stores the coefficients of the PDE. This script is then used to merge
these solution data into the existing *.hdf5 file.

Note: It is recommended to use the more convenient argument '--num_sol_buffer'
that directly save all data into the *.hdf5 file during data generation.
This script will not be used in this case.
"""
import argparse
import time
import shutil
import numpy as np
import h5py


def get_cli_args() -> argparse.Namespace:
    r""" Parse the command-line arguments. """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_npy", "-n", type=int, default=50,
                        help="Number of *.npy files for each PDE variable in "
                        "each *.hdf5 dataset.")
    parser.add_argument("--sol_keys", "-k", type=str, default="u",
                        help="Names of the solution variables, seperated by "
                        "','. Examples: 'u0,u1'. Default: 'u'.")
    parser.add_argument("--file_list", "-f", type=str, nargs="+",
                        help="*.hdf5 files to be processed.")
    parser.add_argument("--move_dir", "-m", type=str, default="",
                        help="Move processed file to target directory. "
                        "Default: '', do not move.")
    args = parser.parse_args()
    return args


def merge_npy(hdf5_path: str,
              num_npy: int = 50,
              sol_keys: str = "u",
              move_dir: str = "") -> None:
    sol_dict = {}
    for key in sol_keys.split(","):
        solution = []
        for j in range(num_npy):
            solution.append(np.load(f"{hdf5_path}_{key}_{j}.npy"))
        sol_dict[key] = np.concatenate(solution, axis=0)
    with h5py.File(hdf5_path + ".hdf5", mode="a") as h5file:
        if "sol" in h5file.keys():
            raise RuntimeError(f"File {hdf5_path}.hdf5 already has solution data!")
        for key, sol_array in sol_dict.items():
            h5file.create_dataset(f"sol/{key}", data=sol_array)
    if move_dir == "":
        return  # do not move files
    shutil.move(hdf5_path + ".hdf5", move_dir)
    for key in sol_keys.split(","):
        for j in range(num_npy):
            shutil.move(f"{hdf5_path}_{key}_{j}.npy", move_dir)


def main() -> None:
    args = get_cli_args()
    for filename in args.file_list:
        basename, ext = filename.rsplit(".", 1)
        if ext != "hdf5":
            continue
        merge_npy(basename, args.num_npy, args.sol_keys, args.move_dir)
        print(time.strftime("%H:%M:%S") + " - Reformed " + filename)
        time.sleep(1)


if __name__ == "__main__":
    main()
