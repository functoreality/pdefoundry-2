# fenicsx 方程求解

安装 Python 依赖：

安装 fenicsx 及扩展 dolfinx_mpc （用于实现周期边界条件）

```bash
conda install -c conda-forge fenics-dolfinx
conda install -c conda-forge dolfinx_mpc
```

安装其他 python 依赖

```bash
pip3 install h5py scipy pytest matplotlib
```

运行脚本示例见 run.sh。由于 dolfinx_mpc 的非线性方程组求解器是未发布版本，存在内存泄露问题，时间步数超过一定值后会导致求解失败，因此建议采取显式时间离散格式。包括

- forward_euler
- ssprk22
- ssprk33
- ssprk104
