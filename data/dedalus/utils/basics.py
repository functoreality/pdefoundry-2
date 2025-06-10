r"""Basic utilities for data generation using Dedalus V3."""
from abc import abstractmethod
from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray
import dedalus.public as d3

from ...common.basics import PDETypeBase


class DedalusPDEType(PDETypeBase):
    r"""
    Generate dataset of 2D time-dependent PDE solutions using Dedalus-v3.
    Abstract base class.
    """
    # Basic PDE information
    SOLVER: str = "dedalus"
    # VERSION: float = 5.1

    # Dedalus Parameters
    TIMESTEPPER = d3.SBDF2
    TRIAL_N_SUB_STEPS: List[int] = [1]

    @abstractmethod
    def get_dedalus_problem(self) -> Tuple:
        r"""
        Define the PDE problem to be solved for Dedalus v3 package.

        Returns:
            var_dict (Dict[str, d3.Field]): Dictionary containing all the
                variables to be saved.
            problem (d3.IVP): Dedalus initial-value problem instance.
        """

    def gen_solution(self) -> None:
        def try_solution(max_timestep: float) -> Dict[str, List[NDArray[float]]]:
            var_dict, problem = self.get_dedalus_problem()
            # var_dict: Dict[str, d3.Field]; problem: d3.IVP

            # Solver
            solver = problem.build_solver(self.TIMESTEPPER)
            # solver.stop_sim_time = self.STOP_SIM_TIME
            # max_timestep = self.STOP_SIM_TIME / 100
            # cfl = d3.CFL(solver, initial_dt=max_timestep, cadence=10, safety=0.5, threshold=0.1,
            #              max_change=1.5, min_change=0.5, max_dt=max_timestep)
            # for u_op in var_dict.values():
            #     cfl.add_velocity(u_op)

            sol_dict = {var_name: [] for var_name in var_dict}
            # Main loop
            # solver.step(1e-6)  # make sol. satisfy div-free cond. etc.
            for t_target in self.coord_dict["t"]:
                t_left = t_target - solver.sim_time
                while t_left > 1e-4:
                    # timestep = cfl.compute_timestep()
                    # if timestep < 1e-4:
                    timestep = min(max_timestep, t_left)
                    solver.step(timestep)
                    t_left = t_target - solver.sim_time
                for var_name, u_op in var_dict.items():
                    u_op.change_scales(1)
                    u_snapshot = np.copy(u_op['g'])
                    sol_dict[var_name].append(u_snapshot)
                    if not np.isfinite(u_snapshot).all():
                        # reject current PDE
                        print(f"failed at sim_time: {solver.sim_time:.4f}")
                        return {}

            return sol_dict

        for t_save_steps in self.TRIAL_N_SUB_STEPS:
            print(f"t_save_steps: {t_save_steps}")
            sol_dict = try_solution(self.STOP_SIM_TIME / (100 * t_save_steps))
            if sol_dict:  # accept non-empty dict
                break

        self.raw_sol_dict = {var_name: np.array(sol_list, dtype=np.float32)
                             for var_name, sol_list in sol_dict.items()}
