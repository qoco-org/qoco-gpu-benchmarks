import cvxpy as cp


def run_clarabel(problem, algebra=None):
    """
    Solve a cvxpy problem using Clarabel solver.
    
    Args:
        problem: cvxpy Problem object
        algebra: Optional string, if "cuda" uses CUCLARABEL solver
    
    Returns:
        dict with keys: setup_time, solve_time, num_iters, objective
    """
    if algebra == "cuda":
        problem.solve(verbose=False, solver="CUCLARABEL")
    else:
        problem.solve(verbose=False, solver="CLARABEL")
    
    setup_time = 0 if problem.solver_stats.setup_time is None else problem.solver_stats.setup_time
    solve_time = problem.solver_stats.solve_time
    num_iters = problem.solver_stats.num_iters if hasattr(problem.solver_stats, 'num_iters') else None
    objective = problem.value
    
    return {
        "setup_time": setup_time,
        "solve_time": solve_time,
        "num_iters": num_iters,
        "objective": objective
    }


def run_qoco(problem, algebra=None):
    """
    Solve a cvxpy problem using QOCO solver.
    
    Args:
        problem: cvxpy Problem object
        algebra: Optional string, if "cuda" uses CUDA algebra backend
    
    Returns:
        dict with keys: setup_time, solve_time, num_iters, objective
    """
    if algebra == "cuda":
        problem.solve(verbose=False, solver="QOCO", algebra="cuda")
    else:
        problem.solve(verbose=False, solver="QOCO")
    
    setup_time = 0 if problem.solver_stats.setup_time is None else problem.solver_stats.setup_time
    solve_time = problem.solver_stats.solve_time
    num_iters = problem.solver_stats.num_iters if hasattr(problem.solver_stats, 'num_iters') else None
    objective = problem.value
    
    return {
        "setup_time": setup_time,
        "solve_time": solve_time,
        "num_iters": num_iters,
        "objective": objective
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
    
    setup_time = 0 if problem.solver_stats.setup_time is None else problem.solver_stats.setup_time
    solve_time = problem.solver_stats.solve_time
    num_iters = problem.solver_stats.num_iters if hasattr(problem.solver_stats, 'num_iters') else None
    objective = problem.value
    
    return {
        "setup_time": setup_time,
        "solve_time": solve_time,
        "num_iters": num_iters,
        "objective": objective
    }

