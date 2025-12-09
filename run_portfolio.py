from typing import Any


import os
import numpy as np
import pandas as pd
import cvxpy as cp
from scipy import sparse
from problems.portfolio import portfolio
from solvers import run_clarabel, run_qoco, run_mosek


def get_problem_size(prob):
    """Calculate problem size as nnz(A) + nnz(P)"""
    data, _, _ = prob.get_problem_data(cp.CLARABEL)
    nnzA = data["A"].nnz
    nnzP = 0
    if "P" in data.keys():
        nnzP = sparse.triu(data["P"], format="csc").nnz
    return nnzP + nnzA


def run_portfolio_benchmarks():
    """
    Run portfolio optimization benchmarks for different values of k
    and write results to CSV files using pandas.
    """
    # Create portfolio directory if it doesn't exist
    os.makedirs("portfolio", exist_ok=True)
    
    # Set fixed random seed for reproducibility
    np.random.seed(42)
    
    k_values = [10, 50, 100, 200, 500]
    solvers = {
        "qoco": lambda prob: run_qoco(prob, algebra=None),
        "qoco_cuda": lambda prob: run_qoco(prob, algebra="cuda"),
        "clarabel": lambda prob: run_clarabel(prob, algebra=None),
        "cuclarabel": lambda prob: run_clarabel(prob, algebra="cuda"),
        "mosek": lambda prob: run_mosek(prob)
    }
    
    # Dictionary to store all results
    results = {solver_name: [] for solver_name in solvers.keys()}
    
    # Track which CUDA solvers have been warmed up
    cuda_warmed_up = set[Any]()
    cuda_solvers = {"qoco_cuda", "cuclarabel"}
    
    # Run benchmarks for each k value
    for k in k_values:
        print(f"Solving portfolio problem for k={k}...")
        prob = portfolio(k)
        # Calculate problem size once per k
        problem_size = get_problem_size(prob)
        
        # For k >= 1000, only run CUDA solvers
        if k > 500:
            active_solvers = {name: func for name, func in solvers.items() 
                            if name in ["cuclarabel", "qoco_cuda"]}
        else:
            active_solvers = solvers
        
        for solver_name, solver_func in active_solvers.items():
            print(f"  Running {solver_name}...")
            try:                
                # Warmup CUDA solvers on first call
                if solver_name in cuda_solvers and solver_name not in cuda_warmed_up:
                    print(f"    Warming up {solver_name} (CUDA initialization)...")
                    _ = solver_func(prob)
                    cuda_warmed_up.add(solver_name)
                
                stats = solver_func(prob)
                stats["size"] = problem_size
                results[solver_name].append(stats)
            except Exception as e:
                print(f"    Error with {solver_name}: {e}")
                # Add error entry
                results[solver_name].append({
                    "size": problem_size,
                    "setup_time": None,
                    "solve_time": None,
                    "num_iters": None,
                    "objective": None,
                    "error": str(e)
                })
    
    # Write results to CSV files using pandas
    for solver_name, solver_results in results.items():
        csv_filename = os.path.join("portfolio", f"{solver_name}_results.csv")
        print(f"Writing {csv_filename}...")
        
        # Create DataFrame from results
        df = pd.DataFrame(solver_results)
        
        # Select columns in desired order
        columns = ["size", "setup_time", "solve_time", "num_iters", "objective"]
        # Only include columns that exist in the dataframe
        columns = [col for col in columns if col in df.columns]
        
        # Write to CSV
        df[columns].to_csv(csv_filename, index=False)
        
        print(f"  Wrote {len(solver_results)} rows to {csv_filename}")


if __name__ == "__main__":
    run_portfolio_benchmarks()

