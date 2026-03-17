import argparse
from typing import Any
import os
from problems.portfolio import *
from problems.huber import *
from problems.group_lasso import *
from problems.multiperiod_portfolio import *
from problems.tv_denoising import *
from solvers import SOLVERS, get_problem_size
from utils import write_results

PROBLEMS = [
    "portfolio",
    "huber",
    "group_lasso",
    "multiperiod_portfolio",
    "tv_denoising",
]

PROB_HANDPARSED = {
    "portfolio": portfolio_handparsed,
    "huber": huber_handparsed,
    "group_lasso": group_lasso_handparsed,
    "multiperiod_portfolio": multiperiod_portfolio_cvxpy,
    "tv_denoising": tv_denoising_cvxpy,
}

PROB_CVXPY = {
    "portfolio": portfolio_cvxpy,
    "huber": huber_cvxpy,
    "group_lasso": group_lasso_cvxpy,
    "multiperiod_portfolio": multiperiod_portfolio_cvxpy,
    "tv_denoising": tv_denoising_cvxpy,
}

PROB_SIZES = {
    "portfolio": [10, 50, 100, 200, 500, 900, 1300, 1800],
    "huber": [50, 200, 500, 1000, 2000, 4000, 6000, 10000],
    "group_lasso": [
        5,
        20,
        50,
        100,
        150,
        300,
        450,
        750,
    ],  # Any larger than 750 and CuClarabel runs out of memory
    "multiperiod_portfolio": [2, 5, 10, 15, 25, 50, 75, 125],
    "tv_denoising": [0, 1, 2, 3, 4, 5, 6, 7],
}

MAX_CPU_SIZE = {
    "portfolio": 1800,
    "huber": 10000,
    "group_lasso": 750,
    "multiperiod_portfolio": 125,
    "tv_denoising": 8,
}


def run_benchmarks(prob_name):
    os.makedirs(prob_name, exist_ok=True)
    n_values = PROB_SIZES[prob_name]
    max_cpu_size = MAX_CPU_SIZE[prob_name]

    # Dict for results
    results = {solver_name: [] for solver_name in SOLVERS.keys()}

    # Track which CUDA solvers have been warmed up
    cuda_warmed_up = set[Any]()
    cuda_solvers = {"qoco_cuda", "cuclarabel"}

    # Run benchmarks for each n value
    for n in n_values:
        print(f"Solving {prob_name} problem for n={n}...")
        prob = PROB_HANDPARSED[prob_name](n)
        prob_cvxpy = PROB_CVXPY[prob_name](n)
        problem_size = get_problem_size(prob)

        # For n > max_cpu_size, only run GPU solvers
        if n > max_cpu_size:
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
            stats["size"] = problem_size
            stats["name"] = prob_name + "_" + str(n)
            results[solver_name].append(stats)

    write_results(results, prob_name)


def main():
    parser = argparse.ArgumentParser(description="Run benchmark problems")
    parser.add_argument(
        "--problems",
        required=True,
        type=str,
        help="Problem name (e.g. huber) or 'all'",
    )
    args = parser.parse_args()

    if args.problems == "all":
        for prob in PROBLEMS:
            run_benchmarks(prob)
    else:
        if args.problems not in PROBLEMS:
            raise ValueError(
                f"Unknown problem '{args.problems}'. "
                f"Available options: {PROBLEMS + ['all']}"
            )
        run_benchmarks(args.problems)


if __name__ == "__main__":
    main()
