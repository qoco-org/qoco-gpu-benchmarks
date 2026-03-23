# qoco-gpu-benchmarks

To run the benchmarks follow the steps

1. Create a python 3.13 virtual environment
2. Run `pip install -r requirements.txt`
3. Install CuClarabel: https://www.cvxpy.org/install/index.html?h=cuclarabel. You may have to install it in the PyCall julia environment. Since I use anaconda this required running the following commands:
```
julia --project=/home/govind/anaconda3/envs/test/julia_env -e 'import Pkg; Pkg.add("CUDA")'
```

```
julia --project=/home/govind/anaconda3/envs/test/julia_env -e 'import Pkg; Pkg.add(Pkg.PackageSpec(
    url="https://github.com/oxfordcontrol/Clarabel.jl",
    rev="CuClarabel"
))'
```
4. Run `./run_all.sh` and the figures will be created in the `figures/` directory
