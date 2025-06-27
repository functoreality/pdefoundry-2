[ENGLISH](README.md) | 简体中文

# PDEFoundry-2：2D PDE 基础模型预训练数据生成

## 概述

偏微分方程（PDE）基础模型的预训练需要多样化的数据。
为了训练针对 2D PDE 的基础模型 PDEformer-2（[GitHub](https://github.com/functoreality/pdeformer-2)，[Gitee](https://gitee.com/functoreality/pdeformer-2)），
我们构建了 PDEFoundry-2 预训练数据集，其中包含不同的 PDE 形式、区域形状、边界条件、变量与方程个数、含时情况。
总共约 40TB 的完整预训练数据集可以使用本仓库 `download/` 目录内提供的脚本下载。
注意，由于这些数据集中的每个样本对应不同的特定 PDE 形式，它并不适用于训练通常意义上的神经算子（例如 FNO、DeepONet）。

本仓库提供了 PDEFoundry-2 预训练数据集的生成代码。
此外，为了验证预训练后模型的性能，我们还额外准备了 Sine-Gordon、INS-Tracer、INS-Pipe 这三个（针对特定 PDE 的）微调数据集，
以及反问题数据集。
这部分的数据同样可以使用 `download/` 目录中的脚本下载，相应的数据生成代码也同样包含在本仓库中。

数据集下载后可以直接使用，无需额外运行本仓库提供的其他脚本文件，也不需要按下文的说明安装 Python 依赖。
只有在需要自己运行数据生成程序的情况下，用户才需要继续阅读本文档的后续内容。

## 安装 Python 依赖

基于谱方法的 Dedalus V3 可以按照 [这里](https://dedalus-project.readthedocs.io/en/latest/pages/installation.html) 的说明安装。
若要使用基于有限元的 FEniCSx，我们还需要额外安装扩展 `dolfinx_mpc` 以支持周期边界条件。使用如下命令安装：

```bash
conda install -c conda-forge fenics-dolfinx
conda install -c conda-forge dolfinx_mpc
pip3 install h5py scipy pytest matplotlib
```

每类数据的生成只依赖于 Dedalus、FEniCSx 之一，不存在同时依赖的情况。
因此，两个软件包并不是必须安装在同一个 conda 环境中，只需要保证运行特定程序的环境已包括相应的依赖就可以了。

## 运行 Dedalus 数据生成程序

Dedalus 安装完成后，可运行相应的脚本文件生成数据。
例如，使用如下命令生成至多带 3 个三角函数项的、随机种子为 1 的 反应-对流-扩散（DCR）方程预训练数据集：

```bash
OMP_NUM_THREADS=1 python3 -m data.dedalus.forward_main.diff_convec_reac -J 3 -r 1
```

（注意不能使用普通的 Python 程序运行命令 `python3 data/dedalus/forward_main/diff_convec_reac.py`）。
可使用如下命令查看命令行参数列表：

```bash
python3 -m data.dedalus.forward_main.diff_convec_reac -h
```

在有较多 CPU 资源的服务器上，可以考虑通过如下命令同时运行多个数据生成程序
（⚠️运行前请确保您充分理解这些命令的含义，确认您能够处理其运行过程可能产生的问题）：

```bash
export OMP_NUM_THREADS=1
function gen_seed() {
	echo $1
	python3 -m data.dedalus.forward_main.diff_convec_reac -r $1 --resume --num_sol_buffer 20
	# for i in {1..101}; do  # 若发现圆盘区域数据生成出现不明原因内存增长，可考虑改用此命令生成
	# 	python3 -m data.dedalus.forward_main.disk_dcr -r $1 --resume --num_sol_buffer 10 --terminate_on_save
	# 	sleep 4
	# done
}
for r in {1..10}; do (gen_seed $r &) && sleep 10; done
gen_seed 11
```

预训练 PDE 类型与生成脚本的对应关系如下表所示：

| PDE 类型 | 解释 | 生成脚本 |
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

微调数据对应的生成命令如下：

```bash
# Sine-Gordon, dedalus_v5.1_Baseline2D_SineGordon_k1e-02_4_seed0.hdf5
OMP_NUM_THREADS=1 python3 -m data.dedalus.forward_main.sine_gordon -r 0
# INS-Tracer, dedalus_v5.1_Baseline2D_INSTracer_icA_noP_db1_nu0.001D0.01_seed0.hdf5
OMP_NUM_THREADS=1 python3 -m data.dedalus.forward_main.ins_coupled_var -r 0 --ignore_p
# INS-Pipe, dedalus_v5.1_Baseline2D_INS_icA_npY_noP_k1e-03L0.01_seed0.hdf5
OMP_NUM_THREADS=1 python3 -m data.dedalus.forward_main.incomp_ns -r 0 --ignore_p --y_wall
```

## 运行 FEniCSx 数据生成程序

## 文件目录结构
