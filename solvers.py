from dataclasses import dataclass
from typing import Any
import qoco


@dataclass
class ProblemData:
    """Data structure containing problem data for direct solver interface."""

    n: int  # number of variables
    m: int  # number of equality constraints
    p: int  # number of inequality constraints
    P: Any  # quadratic form matrix (sparse)
    c: Any  # linear term (array)
    A: Any  # equality constraint matrix (sparse)
    b: Any  # equality constraint RHS (array)
    G: Any  # inequality constraint matrix (sparse)
    h: Any  # inequality constraint RHS (array)
    l: int  # lower bounds
    nsoc: int  # number of second-order cones
    q: list  # SOC dimensions


def run_clarabel(problem, algebra=None):
    """
    Solve a cvxpy problem using Clarabel solver.

    Args:
        problem: cvxpy Problem object or ProblemData struct
        algebra: Optional string, if "cuda" uses CUCLARABEL solver

    Returns:
        dict with keys: setup_time, solve_time, num_iters, objective
    """
    # Check if problem is a ProblemData struct or cvxpy Problem
    if isinstance(problem, ProblemData):
        # Direct interface not implemented yet for Clarabel
        raise NotImplementedError("Direct interface for Clarabel not yet implemented")

    # Original cvxpy interface
    if algebra == "cuda":
        problem.solve(verbose=False, solver="CUCLARABEL")
    else:
        problem.solve(verbose=False, solver="CLARABEL")

    setup_time = (
        0
        if problem.solver_stats.setup_time is None
        else problem.solver_stats.setup_time
    )
    solve_time = problem.solver_stats.solve_time
    num_iters = (
        problem.solver_stats.num_iters
        if hasattr(problem.solver_stats, "num_iters")
        else None
    )
    objective = problem.value

    return {
        "setup_time": setup_time,
        "solve_time": solve_time,
        "num_iters": num_iters,
        "objective": objective,
    }


def run_qoco(problem, algebra=None):
    """
    Solve a cvxpy problem or direct data using QOCO solver.

    Args:
        problem: cvxpy Problem object or ProblemData struct
        algebra: Optional string, if "cuda" uses CUDA algebra backend

    Returns:
        dict with keys: setup_time, solve_time, num_iters, objective
    """
    # Check if problem is a ProblemData struct or cvxpy Problem
    if isinstance(problem, ProblemData):
        # Direct interface
        algebra_arg = algebra if algebra == "cuda" else "builtin"
        prob = qoco.QOCO(algebra=algebra_arg)
        prob.setup(
            problem.n,
            problem.m,
            problem.p,
            problem.P,
            problem.c,
            problem.A,
            problem.b,
            problem.G,
            problem.h,
            problem.l,
            problem.nsoc,
            problem.q,
            verbose=True,
        )
        res = prob.solve()

        return {
            "setup_time": res.setup_time_sec,
            "solve_time": res.solve_time_sec,
            "num_iters": res.iters,
            "objective": res.obj,
        }

    # Original cvxpy interface
    if algebra == "cuda":
        problem.solve(verbose=True, solver="QOCO", algebra="cuda")
    else:
        problem.solve(verbose=True, solver="QOCO")

    setup_time = (
        0
        if problem.solver_stats.setup_time is None
        else problem.solver_stats.setup_time
    )
    solve_time = problem.solver_stats.solve_time
    num_iters = (
        problem.solver_stats.num_iters
        if hasattr(problem.solver_stats, "num_iters")
        else None
    )
    objective = problem.value

    return {
        "setup_time": setup_time,
        "solve_time": solve_time,
        "num_iters": num_iters,
        "objective": objective,
    }


def run_mosek(problem):
    """
    Solve a cvxpy problem using MOSEK solver.

    Args:
        problem: cvxpy Problem object

    Returns:
        dict with keys: setup_time, solve_time, num_iters, objective
    """
    problem.solve(verbose=False, solver="MOSEK")

    setup_time = (
        0
        if problem.solver_stats.setup_time is None
        else problem.solver_stats.setup_time
    )
    solve_time = problem.solver_stats.solve_time
    num_iters = (
        problem.solver_stats.num_iters
        if hasattr(problem.solver_stats, "num_iters")
        else None
    )
    objective = problem.value

    return {
        "setup_time": setup_time,
        "solve_time": solve_time,
        "num_iters": num_iters,
        "objective": objective,
    }
