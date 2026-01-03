from typing import Any
import os
import numpy as np
from solvers import SOLVERS
from utils import write_results
import h5py
from pathlib import Path
import cvxpy as cp
from scipy import sparse


problem_types = ["huber", "lasso"]
skip = [
    "Springer_ESOC",
    "Rucci_Rucci1",
    "Bates_sls",
]


def run_suitesparse_benchmarks():
    os.makedirs("suitesparse", exist_ok=True)

    # Dict for results
    results = {solver_name: [] for solver_name in SOLVERS.keys()}

    # Track which CUDA solvers have been warmed up
    cuda_warmed_up = set[Any]()
    cuda_solvers = {"qoco_cuda", "cuclarabel"}

    directory = Path("data/suitesparse")
    for file_path in directory.iterdir():
        for problem_type in problem_types:
            if file_path.is_file():
                f = h5py.File(file_path, "r")

                problem_name = file_path.stem + "_" + problem_type
                print(problem_name)

                # Set up CVXPY problem.
                Ax = f["A"]["data"][:]
                Ai = f["A"]["ir"][:]
                Ap = f["A"]["jc"][:]
                b = f["b"][:]

                n = len(Ap) - 1
                m = len(b)
                A = sparse.csc_matrix((Ax, Ai, Ap), shape=(m, n))
                f.close()
                if file_path.stem in skip:
                    continue
                x = cp.Variable(n)
                obj = 0
                if problem_type == "huber":
                    obj = cp.sum(cp.huber(A @ x - b))
                elif problem_type == "lasso":
                    lam = np.linalg.norm(A.T @ b, np.inf)
                    obj = cp.sum_squares(A @ x - b) + lam * cp.norm(x, 1)
                else:
                    raise ValueError
                prob = cp.Problem(cp.Minimize(obj), [])

                active_solvers = SOLVERS
                for solver_name, solver_func in active_solvers.items():
                    print(f"  Running {solver_name}...")

                    # Warmup CUDA solvers on first call
                    if (
                        solver_name in cuda_solvers
                        and solver_name not in cuda_warmed_up
                    ):
                        print(f"    Warming up {solver_name} (CUDA initialization)...")
                        _ = solver_func(prob)
                        cuda_warmed_up.add(solver_name)
                    stats = solver_func(prob)
                    results[solver_name].append(stats)

    write_results(results)


if __name__ == "__main__":
    run_suitesparse_benchmarks()
