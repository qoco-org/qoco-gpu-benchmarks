# QOCO GPU Benchmarks

This repository was used to generated the numerical results in the paper titled [QOCO-GPU: A Quadratic Objective Conic Optimizer with GPU Acceleration](https://arxiv.org/abs/2603.29197).

To run the benchmarks follow the steps

1. Create a python 3.13 virtual environment
2. Run `pip install -r requirements.txt`
3. Install CuClarabel: https://www.cvxpy.org/install/index.html?h=cuclarabel. You may have to install it in the PyCall julia environment. Since I use anaconda this required running the following commands:

   First, pin `juliacall` to a version whose bundled `PythonCall` matches what the
   CuClarabel branch requires (`PythonCall = "=0.9.31"`). Newer `juliacall`
   releases pin `PythonCall` to a different version and make the Julia environment
   unsolvable, which causes the `PythonExt` extension (and `cupy_to_cucsrmat`) to be
   missing at runtime:
```
pip install 'juliacall==0.9.31'
python -c 'import juliacall'   # re-resolves the Julia env with PythonCall 0.9.31
```

```
julia --project=/home/govind/anaconda3/envs/test/julia_env -e 'import Pkg; Pkg.add(Pkg.PackageSpec(
    url="https://github.com/oxfordcontrol/Clarabel.jl",
    rev="CuClarabel"
))'
```

```
julia --project=/home/govind/anaconda3/envs/test/julia_env -e 'import Pkg; Pkg.add("CUDA")'
```
4. Run `./run_all.sh` and the figures will be created in the `figures/` directory

## Citing
```
@article{chari2026qocogpu,
  title = {{QOCO}-{GPU}: A Quadratic Objective Conic Optimizer with GPU Acceleration},
  author = {Chari, Govind M and A{\c{c}}{\i}kme{\c{s}}e, Beh{\c{c}}et},
  journal = {arXiv preprint arXiv:2603.29197},
  year = {2026},
}
```
