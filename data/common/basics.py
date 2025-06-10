r""" Basic utilities utilized during data generation. """
import time
import logging
import argparse
import os
import sys
from abc import ABC, abstractmethod
import pickle
import hashlib
import json
from typing import Dict, List, Tuple, Union, Optional, Callable

import h5py
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def prepare_plot_2d(field_2d: NDArray[float],
                    coords: Optional[Tuple[NDArray[float]]] = None,
                    ax_labels: str = "xy",
                    title: str = "") -> None:
    r""" Prepare a 2D plot of the field. """
    plt.figure(figsize=(6, 5))
    if coords is None:
        # plt.imshow(field_2d.T, cmap="jet", origin="lower")
        plt.pcolormesh(field_2d.T, cmap="jet")
    else:
        (x_coord, y_coord) = coords
        plt.pcolormesh(x_coord.flat, y_coord.flat, field_2d.T, cmap="jet")
        # shading='gouraud', rasterized=True)
    plt.xlabel(ax_labels[0])
    plt.ylabel(ax_labels[1])
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()


def prepare_plot_2d_video(field_2d: NDArray[float],
                          coords: Optional[Tuple[NDArray[float]]] = None,
                          ax_labels: str = "xy",
                          title: str = "") -> None:
    r""" Prepare a 2D video plot of the time-dependent field. """
    fig, ax = plt.subplots(figsize=(6, 5))
    if coords is None:
        field_2d = field_2d.transpose((0, 2, 1))  # Shape [n_t, n_y, n_x].
        my_im = plt.pcolormesh(field_2d[0], cmap="jet")
    else:
        (x_coord, y_coord) = coords
        if np.ndim(x_coord) == 1 and np.ndim(y_coord) == 1:
            x_coord, y_coord = np.meshgrid(x_coord, y_coord, indexing="ij")
        else:
            x_coord, y_coord = np.broadcast_arrays(x_coord, y_coord)
        my_im = plt.pcolormesh(
            x_coord, y_coord, field_2d[0], cmap="jet")
    plt.clim(np.min(field_2d), np.max(field_2d))
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    def update(frame):
        my_im.set_array(field_2d[frame])
        ax.set_title(f"{title} (T={frame})")
    update(0)
    frames = np.linspace(0, len(field_2d) - 1, 25 + 1, dtype=int)
    anim = FuncAnimation(fig, update, frames=frames, interval=100)

    plt.xlabel(ax_labels[0])
    plt.ylabel(ax_labels[1])
    plt.colorbar()
    plt.tight_layout()
    return anim


class PDETermBase(ABC):
    r""" abstract class for special PDE terms used in PDEDataGenBase class """

    def __str__(self) -> str:
        data_dict = self.get_data_dict(" ")
        str_list = [f"{key}: {data}" for key, data in data_dict.items()]
        return " ".join(str_list)

    @staticmethod
    def add_cli_args_(parser: argparse.ArgumentParser) -> None:
        r""" Add command-line arguments for this PDE term. """

    @staticmethod
    def arg_str(args: argparse.Namespace) -> str:
        r"""
        Obtain string representation of the command-line arguments to be used
        in data file names.
        """
        return ""

    @abstractmethod
    def reset(self, rng: np.random.Generator) -> None:
        r"""
        Reset the random parameters in the current term.
        Args:
            rng (numpy.random.Generator): Random number generator instance.
        """

    @abstractmethod
    def reset_debug(self) -> None:
        r""" Reset the parameters in the current term to its default value. """

    def gen_dedalus_ops(self):
        r""" Generate the operator for the Dedalus solver. """
        raise NotImplementedError

    @abstractmethod
    def get_data_dict(self, prefix: str
                      ) -> Dict[str, Union[int, float, NDArray]]:
        r"""
        Returning a dictionary containing the current parameters in this PDE
        term.
        """
        return {}

    def prepare_plot(self, title: str = "") -> None:
        r"""
        Create a matplotlib figure showing the current parameters in this PDE
        term. Use `plt.show()` afterwards to see the results.
        """


class PDETypeBase(ABC):
    r"""
    Generate dataset of 1D time-dependent PDE solutions. Abstract base class.
    """
    # Basic PDE information
    VERSION: float = 5.1
    SOLVER: str = "dedalus"
    PREPROCESS_DAG: bool = False
    PDE_TYPE_ID: int = 0
    IS_INVERSE: bool = False
    n_vars: int = 1  # number of components in the PDE

    STOP_SIM_TIME = 1.
    TRIAL_N_SUB_STEPS: List[int] = [1]  # mainly for Dedalus

    # solution data
    term_obj_dict: Dict[str, PDETermBase]
    coord_dict: Dict[str, NDArray[float]]
    raw_sol_dict: Dict[str, NDArray[float]]

    def __init__(self, args: argparse.Namespace) -> None:
        self.u_bound = args.u_bound
        self.coord_dict = {"t": np.linspace(0, self.STOP_SIM_TIME, 101)}
        self.term_obj_dict = {}

    @property
    def coef_dict(self) -> Dict[str, Union[int, float, NDArray]]:
        r""" A dictionary containing the current PDE coefficients. """
        coef_dict = {}
        for prefix, term_obj in self.term_obj_dict.items():
            coef_dict.update(term_obj.get_data_dict(prefix))
        return coef_dict

    @property
    def coef_str_dict(self) -> Dict[str, str]:
        r"""
        A dictionary containing the string representation of the current PDE
        coefficients.
        """
        str_dict = {prefix: str(term_obj)
                    for prefix, term_obj in self.term_obj_dict.items()}
        return str_dict

    @property
    def pde_info_dict(self) -> Dict[str, Union[bool, int, float]]:
        r"""A dictionary containing some information about the PDE."""
        return {"version": self.VERSION, "preprocess_dag": self.PREPROCESS_DAG,
                "pde_type_id": self.PDE_TYPE_ID, "is_inverse": self.IS_INVERSE,
                "trial_n_sub_steps": np.array(self.TRIAL_N_SUB_STEPS)}

    @property
    def sol_dict(self) -> Dict[str, NDArray[float]]:
        r"""
        A dictionary containing the post-processed PDE solutions that will be
        saved into the dataset. The default behavior is to return the solutions
        generated by Dedalus (when no post-processing is required), and users
        may override this method depending on their need.
        """
        return self.raw_sol_dict

    @staticmethod
    @abstractmethod
    def get_hdf5_file_prefix(args: argparse.Namespace) -> str:
        r""" Get the file prefix for the resulting HDF5 data file. """
        return "unnamed_pde"

    @classmethod
    def get_cli_args_parser(cls) -> argparse.ArgumentParser:
        r""" Get the command-line argument parser. """
        parser = argparse.ArgumentParser(description=cls.__doc__)
        return parser

    def reset_pde(self, rng: np.random.Generator) -> None:
        r""" Randomly reset the terms in the current PDE. """
        self.raw_sol_dict = {}
        for term_obj in self.term_obj_dict.values():
            term_obj.reset(rng)

    def reset_sample(self, rng: np.random.Generator) -> None:
        r"""
        Reset the current sample of the current PDE for inverse problem. Needs
        to be overriden only for inverse problems.
        """
        if self.IS_INVERSE:
            raise NotImplementedError(
                "Method 'reset_sample' should be overriden "
                "when generating inverse problem dataset.")
        self.reset_pde(rng)

    def reset_debug(self) -> None:
        r""" Reset the terms in the current PDE to the default status. """
        self.raw_sol_dict = {}
        for term_obj in self.term_obj_dict.values():
            term_obj.reset_debug()

    def print_current_coef(self,
                           print_fn: Union[None, Callable] = print,
                           print_coef_level: int = 1) -> None:
        r""" Print the current coefficients of the PDE. """
        if not (print_coef_level > 0 and callable(print_fn)):
            return
        if print_coef_level in [1, 3]:
            print_str = "coefs: "
            for prefix, coef_str in self.coef_str_dict.items():
                print_str += f"\n  {prefix}: {coef_str}"
            print_fn(print_str)
        if print_coef_level in [2, 3]:
            print_str = "raw coefs: "
            for key, data in self.coef_dict.items():
                if np.size(data) < 20:
                    print_str += f"\n  {key}: {data}"
            print_fn(print_str)

    @abstractmethod
    def gen_solution(self) -> None:
        r"""
        Generate the PDE solution corresponding to the current PDE parameters.
        The solution need to be stored in `self.raw_sol_dict`.
        """

    def accept_sol(self, print_fn: Optional[Callable] = None) -> bool:
        r"""Decide whether the current solution should be accepted."""
        if not self.raw_sol_dict:  # dict empty
            if callable(print_fn):
                print_fn("Failed to generate solution.")
            return False
        for var_name, sol_arr in self.raw_sol_dict.items():
            if not np.isfinite(sol_arr).all():
                if callable(print_fn):
                    print_fn(f"rejected: {var_name} is not finite")
                return False
            u_max = np.max(np.abs(sol_arr))
            if u_max > self.u_bound:
                if callable(print_fn):
                    print_fn(f"rejected: {var_name} max {u_max:.2f}")
                return False
        return True

    def plot(self, plot_coef: bool = True) -> None:
        r"""
        Plot the current PDE solution as well as the coefficients (optional).
        """
        # plot coefficients
        if plot_coef:
            for prefix, term_obj in self.term_obj_dict.items():
                term_obj.prepare_plot(prefix)

        # plot current solution
        anim_list = []  # keep animation in variables to prevent deletion
        for var_name, sol_arr in self.sol_dict.items():
            sol_arr = np.squeeze(sol_arr)
            if np.ndim(sol_arr) == 1 and "t" in self.coord_dict:
                plt.figure()
                plt.plot(self.coord_dict["t"], sol_arr)  # 0D (scalar) case
                plt.title(var_name)
                continue

            if "x" not in self.coord_dict:
                raise KeyError(
                    "Please specify coord_dict['x'] on class initialization")
            if "y" in self.coord_dict:  # 2D case
                coords = (self.coord_dict["x"], self.coord_dict["y"])
                anim_list.append(prepare_plot_2d_video(
                    sol_arr, coords=coords, title=var_name))
            elif "t" in self.coord_dict:  # 1D case
                coords = (self.coord_dict["t"], self.coord_dict["x"])
                prepare_plot_2d(sol_arr, coords, "tx", title=var_name)

        plt.show()


class PDEDataRecorder:
    r""" Collect all generated PDE solutions. """
    coef_dict: Dict[str, List[Union[int, float, NDArray, List]]]
    sol_dict: Dict[str, List[NDArray[float]]]
    pde_info_dict: Dict[str, Union[bool, int, float]]
    coord_dict: Dict[str, Union[float, NDArray[float]]]

    def __init__(self, n_data_target: int) -> None:
        self.n_data_target = n_data_target
        self.n_data_current = 0
        self.n_data_saved = 0

    def __len__(self) -> int:
        return self.n_data_current

    @property
    def n_data_left(self) -> int:
        r""" Number of data samples that remain to be generated. """
        return self.n_data_target - self.n_data_current

    @property
    def n_data_buffer(self) -> int:
        r""" Number of data samples in buffer. """
        return self.n_data_current - self.n_data_saved

    def init_dict(self, pde_data_obj) -> None:
        r""" Initialize the dictionary for storing the PDE solutions. """
        self.coef_dict = {key: [] for key in pde_data_obj.coef_dict}
        self.sol_dict = {key: [] for key in pde_data_obj.sol_dict}
        self.pde_info_dict = pde_data_obj.pde_info_dict
        self.coord_dict = pde_data_obj.coord_dict

    def record_data(self, pde_data_obj) -> None:
        r"""
        Record the current PDE solution data.

        Args:
            pde_data_obj (Union[PDETypeBase, PDEDataRecorder])
        """
        if self.n_data_buffer == 0:  # init record
            self.init_dict(pde_data_obj)
        for key, value in pde_data_obj.coef_dict.items():
            self.coef_dict[key].append(value)
        for key, value in pde_data_obj.sol_dict.items():
            self.sol_dict[key].append(value)
        self.n_data_current += 1

    def save_batch_sol_h5(self,
                          args: argparse.Namespace,
                          tmp_filepath: Optional[str] = None) -> None:
        r""" Save a batch of generated solutions to an HDF5 file. """
        if tmp_filepath is None:
            return  # do not save dynamically
        if self.n_data_buffer < args.num_sol_buffer and self.n_data_left > 0:
            return  # saving threshold not reached
        if self.n_data_current <= args.num_sol_buffer:
            # first batch, create the temporary file
            os.makedirs(os.path.dirname(tmp_filepath), exist_ok=True)
            with h5py.File(tmp_filepath, "w") as h5_file:
                for key, value in self.pde_info_dict.items():
                    h5_file.create_dataset("pde_info/" + key, data=value)
                for key, value in vars(args).items():
                    if isinstance(value, (int, float, np.ndarray)):
                        h5_file.create_dataset("args/" + key, data=value)
                for key, value in self.coord_dict.items():
                    h5_file.create_dataset("coord/" + key, data=value)
                for key, data_list in self.coef_dict.items():
                    datashape = (args.num_pde,) + np.array(data_list[0]).shape
                    datatype = np.array(data_list[0]).dtype
                    h5_file.create_dataset("coef/" + key, datashape,
                                           dtype=datatype)
                for key, data_list in self.sol_dict.items():
                    datashape = (args.num_pde,) + np.array(data_list[0]).shape
                    datatype = np.array(data_list[0]).dtype
                    h5_file.create_dataset("sol/" + key, datashape,
                                           dtype=datatype)

        if not os.path.exists(tmp_filepath):
            raise FileNotFoundError(f"Temporary file {tmp_filepath} not found.")

        with h5py.File(tmp_filepath, "a") as h5_file:
            start = self.n_data_saved
            end = self.n_data_current
            for key, data_list in self.coef_dict.items():
                h5_file["coef/" + key][start:end] = np.array(data_list)
                self.coef_dict[key] = []
            for key, data_list in self.sol_dict.items():
                h5_file["sol/" + key][start:end] = np.array(data_list)
                self.sol_dict[key] = []
        self.n_data_saved += end - start

    def save_batch_sol_npy(self,
                           args: argparse.Namespace,
                           filename: str) -> None:
        r""" Save a batch of generated solutions to a NumPy file. """
        if args.num_sol_per_npy <= 0:
            return  # do not save solutions to *.npy files
        if self.n_data_buffer < args.num_sol_per_npy and self.n_data_left > 0:
            return  # saving threshold not reached
        batch_size = 0
        for key in self.coef_dict:
            batch_size = max(batch_size, len(self.coef_dict[key]))
            npy_idx = (self.n_data_current - 1) // args.num_sol_per_npy
            npy_suffix = os.path.join("coef", f"{key}_{npy_idx}.npy")
            npy_filepath = os.path.join(args.h5_file_dir, "npy", filename, npy_suffix)
            os.makedirs(os.path.dirname(npy_filepath), exist_ok=True)
            np.save(npy_filepath, np.array(self.coef_dict[key]))
            self.coef_dict[key] = []
        for key in self.sol_dict:
            batch_size = max(batch_size, len(self.sol_dict[key]))
            npy_idx = (self.n_data_current - 1) // args.num_sol_per_npy
            npy_suffix = os.path.join("solution", f"{key}_{npy_idx}.npy")
            npy_filepath = os.path.join(args.h5_file_dir, "npy", filename, npy_suffix)
            os.makedirs(os.path.dirname(npy_filepath), exist_ok=True)
            np.save(npy_filepath, np.array(self.sol_dict[key]))
            self.sol_dict[key] = []
        self.n_data_saved += batch_size

    def save_hdf5(self,
                  args: argparse.Namespace,
                  filename: str,
                  tmp_filepath: Optional[str] = None) -> None:
        r""" Save the generated data to the target HDF5 file. """
        h5_filepath = os.path.join(args.h5_file_dir, filename + ".hdf5")

        if tmp_filepath is not None:
            os.makedirs(os.path.dirname(h5_filepath), exist_ok=True)
            os.rename(tmp_filepath, h5_filepath)
            return

        with h5py.File(h5_filepath, "w") as h5_file:
            for key, value in self.pde_info_dict.items():
                h5_file.create_dataset("pde_info/" + key, data=value)
            for key, value in vars(args).items():
                if isinstance(value, (int, float, np.ndarray)):
                    h5_file.create_dataset("args/" + key, data=value)
            for key, value in self.coord_dict.items():
                h5_file.create_dataset("coord/" + key, data=value)
            for key, data_list in self.coef_dict.items():
                if not data_list:  # list empty; occurs when saved as npy
                    continue
                h5_file.create_dataset("coef/" + key, data=np.array(data_list))
            for key, data_list in self.sol_dict.items():
                if not data_list:  # list empty; occurs when saved as npy
                    continue
                h5_file.create_dataset("sol/" + key, data=np.array(data_list))


def get_cli_args(pde_cls) -> argparse.Namespace:
    r""" Parse the command-line arguments for specific PDE type. """
    parser = pde_cls.get_cli_args_parser()
    parser.add_argument("--u_bound", type=float, default=10,
                        help="accept only solutions with |u| <= u_bound")
    if pde_cls.IS_INVERSE:
        parser.add_argument("--num_pde", "-n", type=int, default=40,
                            help="number of PDEs to generate")
        parser.add_argument("--num_sample_per_pde", "-s", type=int, default=25,
                            help="number of samples for each PDE")
    else:
        parser.add_argument("--num_pde", "-n", type=int, default=1000,
                            help="number of PDEs to generate")
        parser.add_argument("--num_sol_buffer", type=int, default=0,
                            help="Number of solutions stored in memory before "
                            "saving to *.hdf5 file. If not positive, save all "
                            "solutions to the final *.hdf5 file at once. Hint: "
                            "This option is prioritized over '--num_sol_per_npy'.")
        parser.add_argument("--num_sol_per_npy", type=int, default=0,
                            help="(Deprecated in favor of '--num_sol_buffer'.) "
                            "Number of solutions stored in each *.npy "
                            "file. If not positive, keep all solutions in the "
                            "final *.hdf5 file instead, which is the default "
                            "behavior. Hint: If this value is set to be "
                            "positive, then after data generation is "
                            "complete, users may use 'merge_npy.py' to merge "
                            "the solutions into the *.hdf5 dataset file.")
        parser.add_argument("--resume", action="store_true", help="Resume from "
                            "stopped data generation. This option is only valid "
                            "when '--num_sol_buffer' is positive.")
        parser.add_argument("--terminate_on_save", action="store_true",
                            help="(Experimental) Terminate the current data "
                            "generation process each time a batch of solution "
                            "is saved, in order to tackle with unreasonable "
                            "and unidentifiable growth of memory consumption "
                            "during running. This option is valid only when "
                            "'--resume' is set.")
        parser.add_argument("--record_failed_samples", action="store_true",
                            help="Record PDE samples that the solver fails "
                            "(instead of succeeded) to solve.")
    parser.add_argument("--np_seed", "-r", type=int, default=-1,
                        help="NumPy random seed. Default: -1, not to specify seed.")
    parser.add_argument("--print_coef_level", type=int, default=0,
                        choices=[0, 1, 2, 3], help=r"""
                        print coefficients of each generated PDE.
                        0: do not print. 1: print human-readable results.
                        2: print raw data. 3: print both human-readable
                        results and raw data. """)
    parser.add_argument("--plot_results", action="store_true",
                        help="show image(s) of each accepted solution")
    parser.add_argument("--h5_file_dir", type=str, default="results")
    args = parser.parse_args()
    return args


def _get_hdf5_file_name(pde_cls,
                        args: argparse.Namespace) -> str:
    r"""
    Get the name of the target HDF5 file where the generated data is
    stored.
    """
    fname = f"{pde_cls.SOLVER}_v{pde_cls.VERSION:g}_"
    if pde_cls.IS_INVERSE:
        fname += "inv_"
    fname += pde_cls.get_hdf5_file_prefix(args)
    if pde_cls.IS_INVERSE:
        if args.num_pde != 40:
            fname += f"_num{args.num_pde}"
        if args.num_sample_per_pde != 25:
            fname += f"_samples{args.num_sample_per_pde}"
    elif args.num_pde != 1000:
        fname += f"_num{args.num_pde}"
    if args.np_seed == -1:
        fname += time.strftime('_%Y-%m-%d-%H-%M-%S')
    else:
        fname += f"_seed{args.np_seed}"
    return fname


def _load_state(rng: np.random.Generator,
                recorder: PDEDataRecorder,
                state_filepath: Optional[str] = None,
                print_fn: Optional[Callable] = None,
                ) -> Tuple[int, np.random.Generator]:
    r"""
    Load the data generation state from the last saved state file.
    """
    if state_filepath is None:
        return 0, rng  # no resume
    if not os.path.exists(state_filepath):
        return 0, rng  # no state file found
    with open(state_filepath, "rb") as state_file:
        state_dict = pickle.load(state_file)
    i_pde = state_dict["i_pde"]
    rng_state = state_dict["rng_state"]
    n_data_saved = state_dict["n_data_saved"]
    recorder.n_data_current = n_data_saved
    recorder.n_data_saved = n_data_saved
    rng.bit_generator.state = rng_state
    if callable(print_fn):
        print_fn(f"resume from trial No. {i_pde}, "
                 f"PDE generated {n_data_saved}/{recorder.n_data_target}")
    return i_pde, rng


def _save_state(rng: np.random.Generator,
                recorder: PDEDataRecorder,
                i_pde: int,
                state_filepath: Optional[str] = None,
                print_fn: Optional[Callable] = None) -> None:
    r"""
    Save the current data generation state to a file.
    """
    if state_filepath is None:
        return
    if recorder.n_data_buffer != 0:
        return  # only update the state when buffer is empty
    state_dict = {"i_pde": i_pde,
                  "rng_state": rng.bit_generator.state,
                  "n_data_saved": recorder.n_data_saved}
    os.makedirs(os.path.dirname(state_filepath), exist_ok=True)
    with open(state_filepath, "wb") as state_file:
        pickle.dump(state_dict, state_file)
    if callable(print_fn):
        print_fn(f"state saved at trial No. {i_pde}, "
                 f"PDE generated {recorder.n_data_saved}/{recorder.n_data_target}")


def _gen_forward_data(args: argparse.Namespace,
                      rng: np.random.Generator,
                      pde_data_obj: PDETypeBase,
                      filename: str,
                      print_fn: Optional[Callable] = None) -> None:
    r"""
    Generate the PDE solution data, until the target number of samples is
    reached.
    """
    i_pde = 0
    tmp_filepath = None
    state_filepath = None
    # generate dataset identifier based on args
    args_dict = vars(args).copy()
    for key, value in args_dict.items():
        if isinstance(value, np.ndarray):
            args_dict[key] = value.tolist()

    args_json = json.dumps(args_dict, sort_keys=True) + filename
    hash_object = hashlib.sha256(args_json.encode())
    hash_value = hash_object.hexdigest()
    if callable(print_fn):
        print_fn(f"temporary file prefix: {hash_value}")

    if args.num_sol_buffer > 0:
        tmp_filepath = os.path.join(args.h5_file_dir, "tmp", f"{hash_value}.hdf5")
    if args.resume and (args.num_sol_buffer > 0 or args.num_sol_per_npy > 0):
        state_filepath = os.path.join(args.h5_file_dir, "tmp", f"{hash_value}"
                                      "_generator_state.pkl")

    recorder = PDEDataRecorder(args.num_pde)
    i_pde, rng = _load_state(rng, recorder, state_filepath, print_fn)

    if recorder.n_data_left == 0:
        print_fn("All data generated.")
        print_fn(f"Remove {state_filepath} for new generation.")
        return

    while recorder.n_data_left > 0:
        i_pde += 1
        if callable(print_fn):
            print_fn(f"trial No. {i_pde}, "
                     f"PDE generated {len(recorder)}/{args.num_pde}")
        pde_data_obj.reset_pde(rng)
        pde_data_obj.print_current_coef(print_fn, args.print_coef_level)
        pde_data_obj.gen_solution()
        if pde_data_obj.accept_sol(print_fn) ^ args.record_failed_samples:
            if args.plot_results:
                pde_data_obj.plot()
            recorder.record_data(pde_data_obj)
            recorder.save_batch_sol_h5(args, tmp_filepath)
            recorder.save_batch_sol_npy(args, filename)
            if (args.terminate_on_save and args.resume
                    and recorder.n_data_buffer == 0 and recorder.n_data_left > 0):
                _save_state(rng, recorder, i_pde, state_filepath, print_fn)
                sys.exit(0)
        _save_state(rng, recorder, i_pde, state_filepath, print_fn)
    recorder.save_hdf5(args, filename, tmp_filepath)


def _gen_inverse_data(args: argparse.Namespace,
                      rng: np.random.Generator,
                      pde_data_obj: PDETypeBase,
                      filename: str,
                      print_fn: Optional[Callable] = None) -> None:
    r"""
    Generate the PDE solution data, until the target number of samples is
    reached.
    """
    i_pde = 0
    max_trial_sample = 5 * args.num_sample_per_pde
    recorder = PDEDataRecorder(args.num_pde)
    while recorder.n_data_left > 0:
        i_pde += 1
        pde_data_obj.reset_pde(rng)
        sample_recorder = PDEDataRecorder(args.num_sample_per_pde)
        for i_sample in range(max_trial_sample):
            if (1 + len(sample_recorder)) / (1 + i_sample) < 0.1:
                break  # discard current PDE with low acceptance ratio
            samples_trial_left = max_trial_sample - i_sample
            if sample_recorder.n_data_left > samples_trial_left:
                break  # discard current PDE
            if callable(print_fn):
                print_fn(f"PDE {i_pde} ({len(recorder)}/{args.num_pde}), "
                         f"sample {i_sample} "
                         f"({len(sample_recorder)}/{args.num_sample_per_pde})")

            pde_data_obj.reset_sample(rng)
            pde_data_obj.print_current_coef(print_fn, args.print_coef_level)
            pde_data_obj.gen_solution()

            if pde_data_obj.accept_sol(print_fn):
                if args.plot_results:
                    pde_data_obj.plot()
                sample_recorder.record_data(pde_data_obj)
            if sample_recorder.n_data_left <= 0:
                recorder.record_data(sample_recorder)
                break  # next PDE
    recorder.save_hdf5(args, filename)


def _create_logger(path: str = "./log.log") -> logging.RootLogger:
    r""" save the logger information to a file specified by `path` """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logfile = path
    file_handler = logging.FileHandler(logfile, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d]"
                                  " - %(levelname)s: %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


class MyLogger:
    r"""
    A simple logger. Not using the standard logger to exclude Dedalus running
    information.
    """

    def __init__(self, path: str = "./log.log") -> None:
        self.path = path

    def info(self, message: str) -> None:
        r""" Manually print message to screen and write into log. """
        message = time.strftime("%Y-%m-%d %H:%M:%S - ") + message
        print(message)
        with open(self.path, "a", encoding="utf8") as log_file:
            log_file.write(message + "\n")


def gen_data(args: argparse.Namespace, pde_data_obj: PDETypeBase) -> None:
    r""" main data generation process """
    # random number generator
    if args.np_seed == -1:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(args.np_seed)
        np.random.seed(args.np_seed)  # for spm1d.rft1d

    # logger
    os.makedirs("log", exist_ok=True)
    fname = _get_hdf5_file_name(type(pde_data_obj), args)
    # logger = _create_logger(os.path.join("log", fname + ".log"))
    logger = MyLogger(os.path.join("log", fname + ".log"))
    logger.info(f"target file: {fname}.hdf5")

    # generate data and save
    os.makedirs(args.h5_file_dir, exist_ok=True)
    if pde_data_obj.IS_INVERSE:
        _gen_inverse_data(args, rng, pde_data_obj, fname, print_fn=logger.info)
    else:
        _gen_forward_data(args, rng, pde_data_obj, fname, print_fn=logger.info)
    logger.info(f"file saved: {fname}.hdf5")
