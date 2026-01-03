from typing import Any
import os
import numpy as np
from problems.portfolio import *
from solvers import SOLVERS
from utils import write_results


def run_portfolio_benchmarks():
    os.makedirs("portfolio", exist_ok=True)
    k_values = [10, 50]

    # Dict for results
    results = {solver_name: [] for solver_name in SOLVERS.keys()}

    # Track which CUDA solvers have been warmed up
    cuda_warmed_up = set[Any]()
    cuda_solvers = {"qoco_cuda", "cuclarabel"}

    # Run benchmarks for each k value
    for k in k_values:
        print(f"Solving portfolio problem for k={k}...")
        prob = portfolio_handparsed(k)
        prob_cvxpy = portfolio_cvxpy(k)

        # For k >= 1000, only run GPU solvers
        if k > 1000:
            active_solvers = {
                name: func
                for name, func in SOLVERS.items()
                if name in ["cuclarabel", "qoco_cuda"]
            }
        else:
            active_solvers = SOLVERS

        for solver_name, solver_func in active_solvers.items():
            print(f"  Running {solver_name}...")

            # Warmup CUDA solvers on first call
            if solver_name in cuda_solvers and solver_name not in cuda_warmed_up:
                print(f"    Warming up {solver_name} (CUDA initialization)...")
                _ = solver_func(prob)
                cuda_warmed_up.add(solver_name)

            if solver_name == "mosek":
                stats = solver_func(prob_cvxpy)
            else:
                stats = solver_func(prob)
            results[solver_name].append(stats)

    write_results(results)


if __name__ == "__main__":
    run_portfolio_benchmarks()
