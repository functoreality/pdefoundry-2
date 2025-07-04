ENGLISH | [简体中文](README_CN.md)

# PDEFoundry-2: Pretraining Dataset for 2D PDE Foundation Models

## Overview

Pretraining a foundation model to solve partial differential equation (PDE) requires diverse data.
To train the foundation model PDEformer-2 for 2D PDEs ([GitHub](https://github.com/functoreality/pdeformer-2), [Gitee](https://gitee.com/functoreality/pdeformer-2)),
we developed the PDEFoundry-2 pretraining dataset, which includes different PDE forms, domain shapes, boundary conditions, number of variables and equations, as well as time-dependency.
The complete pretraining dataset of approximately 40TB can be downloaded using the scripts in the `download/` directory.
Since each sample in this dataset corresponds to a different specific PDE form, it is not suitable for training typical neural operators (such as FNO and DeepONet).

Please note that the dataset contains a small number of samples with non-physical patterns generated by the solver, such as checkerboard-like high-frequency oscillations.
This is because traditional solvers often require parameter tuning tailored to each specific equation,
while the PDEFoundry-2 pretraining dataset exhibits high diversity and significant variations among samples,
making it difficult to preset a single set of solver parameters suitable for all equations.
Additionally, we were unable to devise a reliable criterion for filtering out these anomalous data points.
The pretraining process of PDEformer-2 can tolerate the presence of some anomalous data.
If you intend to use this dataset for other purposes, please exercise caution.

This repository provides the data generation code for the PDEFoundry-2 pretraining dataset.
In addition to this, to validate the performance of the pretrained model, we also prepared three additional fine-tuning datasets (Sine-Gordon, INS-Tracer, INS-Pipe) for specific PDEs,
as well as inverse problem datasets.
These dataset can also be downloaded using scripts in `download/`, and
the corresponding data generation code is included in the repository as well.

The datasets can be used directly after downloading.
It is not necessary to run additional script files provided in this repository or to install Python dependencies as described below.
The remaining content of this document is only provided for readers that intend to run the data generation program themselves.

## Installing Python Dependencies

The Dedalus V3 package based on spectral methods can be installed according to the instructions [here](https://dedalus-project.readthedocs.io/en/latest/pages/installation.html).
To use FEniCSx based on finite element method, we also need to install an additional extension `dolfinx_mpc` to support periodic boundary conditions. Use the following command to install them:

```bash
conda install -c conda-forge fenics-dolfinx
conda install -c conda-forge dolfinx_mpc
pip3 install h5py scipy pytest matplotlib
```

The generation of each type of data depends only on one of Dedalus and FEniCSx, and there is no simultaneous dependence.
Therefore, the two software packages do not have to be installed in the same conda environment.
Just make sure that the environment in which the specific program is running includes the corresponding dependencies.

## Running Dedalus Data Generation

After Dedalus is installed, we can execute the corresponding script file to generate data.
For example, use the following command to generate a diffusion-convection-reaction (DCR) equation pretraining dataset with up to 3 trigonometric terms and a random seed of 1:

```bash
OMP_NUM_THREADS=1 python3 -m data.dedalus.forward_main.diff_convec_reac -J 3 -r 1
```

(Note that you cannot use the command for regular Python programs `python3 data/dedalus/forward_main/diff_convec_reac.py`).
The following command provides a list of the command line arguments:

```bash
python3 -m data.dedalus.forward_main.diff_convec_reac -h
```

On servers with abundant CPU resources, it is possible to run multiple data generation programs in parallel using the following command
(⚠️ please ensure that you fully understand the meaning of these commands, and can handle any issues that may arise during execution):

```bash
export OMP_NUM_THREADS=1
function gen_seed() {
	echo $1
	python3 -m data.dedalus.forward_main.diff_convec_reac -r $1 --resume --num_sol_buffer 20
	# If you observe an unreasonable memory growth during disk-domain data generation, consider switching to the following commands.
	# for i in {1..101}; do
	# 	python3 -m data.dedalus.forward_main.disk_dcr -r $1 --resume --num_sol_buffer 10 --terminate_on_save
	# 	sleep 4
	# done
}
for r in {1..10}; do (gen_seed $r &) && sleep 10; done
gen_seed 11
```

The following table shows the correspondence between pretraining PDE types and the data generation scripts:

| PDE Type | Explanation | Generation Script |
|:--:|:--:|:--:|
| DCR | diffusion-convection-reaction | `diff_convec_reac` |
| DCR | diffusion-convection-reaction | `disk_dcr` |
| Wave | (generalized) wave equation | `wave` |
| Wave | (generalized) wave equation | `disk_wave` |
| MV-DCR | multi-variable DCR | `multi_component` |
| DC-DCR | divergence-constrained DCR | `div_constr_dcr` |
| MV-Wave | multi-variable Wave | `multi_component_wave` |
| DC-Wave | divergence-constrained Wave | `div_constr_wave` |
| G-SWE | generalized shallow-water equation | `shallow_water` |

Data generation commands for the finetuning datasets:

```bash
# Sine-Gordon, dedalus_v5.1_Baseline2D_SineGordon_k1e-02_4_seed0.hdf5
OMP_NUM_THREADS=1 python3 -m data.dedalus.forward_main.sine_gordon -r 0
# INS-Tracer, dedalus_v5.1_Baseline2D_INSTracer_icA_noP_db1_nu0.001D0.01_seed0.hdf5
OMP_NUM_THREADS=1 python3 -m data.dedalus.forward_main.ins_coupled_var -r 0 --ignore_p
# INS-Pipe, dedalus_v5.1_Baseline2D_INS_icA_npY_noP_k1e-03L0.01_seed0.hdf5
OMP_NUM_THREADS=1 python3 -m data.dedalus.forward_main.incomp_ns -r 0 --ignore_p --y_wall
```
