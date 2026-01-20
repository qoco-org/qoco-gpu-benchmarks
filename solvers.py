from dataclasses import dataclass
from typing import Any
import qoco
import scipy.sparse as sp
import numpy as np
import clarabel
import cvxpy as cp
import gurobipy as gp
from gurobipy import GRB

SOLVERS = {
    # "qoco": lambda prob: run_qoco(prob, algebra=None),
    "qoco_cuda": lambda prob: run_qoco(prob, algebra="cuda"),
    # "clarabel": lambda prob: run_clarabel(prob, algebra=None),
    "cuclarabel": lambda prob: run_clarabel(prob, algebra="cuda"),
    "gurobi": lambda prob: run_gurobi(prob),
    # "mosek": lambda prob: run_mosek(prob),
}

VERBOSE = True
TOLERANCE = 1e-7
TIME_LIMIT = 3600


def get_problem_size(prob):
    """Calculate problem size as nnz(A) + nnz(P)"""

    if isinstance(prob, ProblemData):
        Pnnz = prob.P.nnz if prob.P is not None else 0
        return Pnnz + prob.A.nnz + prob.G.nnz
    data, _, _ = prob.get_problem_data(cp.CLARABEL)
    nnzA = data["A"].nnz
    nnzP = 0
    if "P" in data.keys():
        nnzP = sp.triu(data["P"], format="csc").nnz
    return nnzP + nnzA


@dataclass
class ProblemData:
    n: int  # number of variables
    m: int  # number of equality constraints
    p: int  # number of inequality constraints
    P: Any  # quadratic cost
    c: Any  # linear cost
    A: Any  # equality constraint matrix
    b: Any  # equality constraint RHS
    G: Any  # inequality constraint matrix
    h: Any  # inequality constraint RHS
    l: int  # lower bounds
    nsoc: int  # number of second-order cones
    q: list  # SOC dimensions


def dims_to_solver_cones(jl, cone_dims):
    jl.seval("""cones = Clarabel.SupportedCone[]""")

    # Zero cone (equality constraints)
    if cone_dims.zero > 0:
        jl.push_b(jl.cones, jl.Clarabel.ZeroConeT(cone_dims.zero))

    # Nonnegative cone (inequality constraints)
    if cone_dims.nonneg > 0:
        jl.push_b(jl.cones, jl.Clarabel.NonnegativeConeT(cone_dims.nonneg))

    # Second-order cones
    for dim in cone_dims.soc:
        jl.push_b(jl.cones, jl.Clarabel.SecondOrderConeT(dim))


def solve_cuclarabel_direct(data):
    import cupy
    from cupyx.scipy.sparse import csr_matrix as cucsr_matrix
    from juliacall import Main as jl

    jl.seval("using Clarabel, LinearAlgebra, SparseArrays")
    jl.seval("using CUDA, CUDA.CUSPARSE")

    # Combine A and G: [A; G]
    if data.A is not None and data.G is not None:
        A_combined = sp.vstack([data.A, data.G], format="csr")
    elif data.A is not None:
        A_combined = data.A.tocsr()
    elif data.G is not None:
        A_combined = data.G.tocsr()
    else:
        # No constraints
        A_combined = sp.csr_matrix((0, data.n))

    # Combine b and h: [b; h]
    if data.b is not None and data.h is not None:
        b_combined = np.concatenate([data.b, data.h])
    elif data.b is not None:
        b_combined = data.b
    elif data.h is not None:
        b_combined = data.h
    else:
        b_combined = np.array([])

    # Get P and q
    if data.P is not None:
        P = sp.triu(data.P).tocsr()
    else:
        P = sp.csr_matrix((data.n, data.n))

    q = data.c

    # Convert to GPU arrays
    Pgpu = cucsr_matrix(P)
    qgpu = cupy.array(q)
    Agpu = cucsr_matrix(A_combined)
    bgpu = cupy.array(b_combined)

    # Convert P to Julia
    if Pgpu.nnz != 0:
        jl.P = jl.Clarabel.cupy_to_cucsrmat(
            jl.Float64,
            int(Pgpu.data.data.ptr),
            int(Pgpu.indices.data.ptr),
            int(Pgpu.indptr.data.ptr),
            *Pgpu.shape,
            Pgpu.nnz,
        )
    else:
        jl.seval(
            f"""
        P = CuSparseMatrixCSR(sparse(Float64[], Float64[], Float64[], {data.n}, {data.n}))
        """
        )

    jl.q = jl.Clarabel.cupy_to_cuvector(jl.Float64, int(qgpu.data.ptr), qgpu.size)
    jl.A = jl.Clarabel.cupy_to_cucsrmat(
        jl.Float64,
        int(Agpu.data.data.ptr),
        int(Agpu.indices.data.ptr),
        int(Agpu.indptr.data.ptr),
        *Agpu.shape,
        Agpu.nnz,
    )
    jl.b = jl.Clarabel.cupy_to_cuvector(jl.Float64, int(bgpu.data.ptr), bgpu.size)

    # Set up cone dimensions
    # Zero cone: data.p (equality constraints)
    # Nonneg cone: data.l (inequality constraints)
    # SOC cones: data.q (second-order cone dimensions)
    class ConeDims:
        def __init__(self, zero, nonneg, soc):
            self.zero = zero
            self.nonneg = nonneg
            self.soc = soc

    cone_dims = ConeDims(zero=data.p, nonneg=data.l, soc=data.q)
    dims_to_solver_cones(jl, cone_dims)

    jl.seval(
        f"""
        settings = Clarabel.Settings(
            direct_solve_method = :cudss,
            tol_gap_abs = {TOLERANCE},
            tol_gap_rel = {TOLERANCE},
            tol_feas    = {TOLERANCE}
        )
        solver = Clarabel.Solver(settings)
        solver = Clarabel.setup!(solver, P, q, A, b, cones)
        Clarabel.solve!(solver)
        """
    )

    setup_time = 0
    solve_time = float(jl.seval("solver.info.solve_time"))
    num_iters = int(jl.seval("solver.solution.iterations"))
    objective = float(jl.seval("solver.solution.obj_val"))
    status = str(jl.seval("solver.solution.status"))

    jl.seval(
        """
    solver = nothing
    P = nothing
    q = nothing
    A = nothing
    b = nothing
    cones = nothing
    settings = nothing
    GC.gc()
    CUDA.reclaim()
    """
    )

    del Pgpu
    del qgpu
    del Agpu
    del bgpu
    cupy.get_default_memory_pool().free_all_blocks()

    return {
        "setup_time": setup_time,
        "status": status,
        "solve_time": solve_time,
        "num_iters": num_iters,
        "objective": objective,
    }


def solve_gurobi_direct(data):
    model = gp.Model("SOCP")
    model.setParam("OutputFlag", VERBOSE)
    model.setParam("BarConvTol", TOLERANCE)
    model.setParam("BarQCPConvTol", TOLERANCE)
    model.setParam("FeasibilityTol", TOLERANCE)
    model.setParam("OptimalityTol", TOLERANCE)
    model.setParam("TimeLimit", TIME_LIMIT)
    model.setParam("Method", 2)
    model.setParam("Crossover", 0)

    n = data.n
    x = model.addMVar(n, lb=-GRB.INFINITY)

    # -------------------------
    # Objective
    # 0.5 x' P x + c' x
    # -------------------------
    P = data.P
    c = data.c
    if P:
        model.setMObjective(0.5 * P, c, 0.0)
    else:
        model.setObjective(c @ x, gp.GRB.MINIMIZE)

    # -------------------------
    # Equality constraints: A x = b
    # -------------------------
    if data.p > 0:
        A = data.A
        b = data.b
        model.addMConstr(A, x, GRB.EQUAL, b)

    # -------------------------
    # Linear inequalities: G x <= h
    # (Non-SOC rows only)
    # -------------------------
    G = data.G
    h = data.h

    if data.l > 0:
        model.addMConstr(G[: data.l, :], x, GRB.LESS_EQUAL, h[: data.l])

    # -------------------------
    # Second-Order Cone Constraints
    # -------------------------
    row = data.l
    for k in range(data.nsoc):
        qk = data.q[k]

        t = model.addVar(
            obj=0,
            name="soc_t_cone_%d" % (k),
            vtype=gp.GRB.CONTINUOUS,
            lb=0,
            ub=gp.GRB.INFINITY,
        )

        u = model.addMVar(
            qk - 1,
            name="soc_x_cone_%d" % (k),
            vtype=gp.GRB.CONTINUOUS,
            lb=-gp.GRB.INFINITY,
            ub=gp.GRB.INFINITY,
        )

        # Vector part
        Gv = G[row + 1 : row + qk, :]
        hv = h[row + 1 : row + qk]

        # Scalar part
        Gt = G[row, :]
        ht = h[row]

        model.addConstr(ht - Gt @ x == t)
        model.addConstr(hv - Gv @ x == u)

        # ||u||_2^2 <= t^2
        model.addConstr(u @ u <= t * t)
        row += qk

    # -------------------------
    # Solve
    # -------------------------
    model.optimize()
    status = "optimal" if model.Status == 2 else model.Status
    return {
        "setup_time": 0.0,
        "status": status,
        "solve_time": model.Runtime,
        "num_iters": model.BarIterCount,
        "objective": model.ObjVal,
    }


def solve_clarabel_direct(data):
    P = data.P.tocsc()
    q = data.c
    A = sp.vstack((data.A, data.G)).tocsc()
    b = np.concatenate((data.b, data.h), axis=0)
    cones = [
        clarabel.ZeroConeT(data.p),
        clarabel.NonnegativeConeT(data.l),
    ]
    for dim in data.q:
        cones.append(clarabel.SecondOrderConeT(dim))

    settings = clarabel.DefaultSettings()
    settings.verbose = VERBOSE
    settings.tol_gap_abs = TOLERANCE
    settings.tol_gap_rel = TOLERANCE
    settings.tol_feas = TOLERANCE
    # settings.direct_solve_method = "qdldl"

    solver = clarabel.DefaultSolver(P, q, A, b, cones, settings)
    sol = solver.solve()
    return {
        "setup_time": 0,
        "status": sol.status,
        "solve_time": sol.solve_time,
        "num_iters": sol.iterations,
        "objective": sol.obj_val,
    }


def run_clarabel(problem, algebra=None):
    # Check if problem is ProblemData, call direct interface
    if isinstance(problem, ProblemData):
        if algebra == "cuda":
            return solve_cuclarabel_direct(problem)
        else:
            return solve_clarabel_direct(problem)

    # Call solvers via cvxpy interface
    if algebra == "cuda":
        problem.solve(verbose=VERBOSE, solver="CUCLARABEL")
    else:
        problem.solve(verbose=VERBOSE, solver="CLARABEL")

    setup_time = (
        0
        if problem.solver_stats.setup_time is None
        else problem.solver_stats.setup_time
    )
    solve_time = problem.solver_stats.solve_time

    return {
        "setup_time": setup_time,
        "status": problem.status,
        "solve_time": solve_time,
        "num_iters": problem.solver_stats.num_iters,
        "objective": problem.value,
    }


def run_qoco(problem, algebra=None):
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
            verbose=VERBOSE,
        )
        res = prob.solve()

        return {
            "setup_time": res.setup_time_sec,
            "status": res.status,
            "solve_time": res.solve_time_sec,
            "num_iters": res.iters,
            "objective": res.obj,
        }

    # Call solvers via cvxpy interface
    if algebra == "cuda":
        problem.solve(verbose=VERBOSE, solver="QOCO", algebra="cuda")
    else:
        problem.solve(verbose=VERBOSE, solver="QOCO")

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
        "status": problem.status,
        "solve_time": solve_time,
        "num_iters": num_iters,
        "objective": objective,
    }


def run_gurobi(problem):
    # Check if problem is a ProblemData struct or cvxpy Problem
    if isinstance(problem, ProblemData):
        # Direct interface
        return solve_gurobi_direct(problem)

    # Call solvers via cvxpy interface
    problem.solve(
        verbose=VERBOSE,
        solver="GUROBI",
        BarConvTol=TOLERANCE,
        BarQCPConvTol=TOLERANCE,
        FeasibilityTol=TOLERANCE,
        OptimalityTol=TOLERANCE,
    )

    setup_time = problem.solver_stats.setup_time or 0
    solve_time = problem.solver_stats.solve_time
    num_iters = (
        problem.solver_stats.num_iters
        if hasattr(problem.solver_stats, "num_iters")
        else None
    )
    objective = problem.value

    return {
        "setup_time": setup_time,
        "status": problem.status,
        "solve_time": solve_time,
        "num_iters": num_iters,
        "objective": objective,
    }


# Can only solve cvxpy problems, not handparsed ones.
def run_mosek(problem):
    if isinstance(problem, ProblemData):
        raise NotImplementedError("Mosek cannot solve with ProblemData")

    try:
        problem.solve(
            verbose=VERBOSE,
            solver="MOSEK",
            mosek_params={
                "MSK_DPAR_INTPNT_CO_TOL_PFEAS": TOLERANCE,
                "MSK_DPAR_INTPNT_CO_TOL_DFEAS": TOLERANCE,
                "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": TOLERANCE,
                "MSK_DPAR_INTPNT_CO_TOL_MU_RED": TOLERANCE,
                # "MSK_DPAR_OPTIMIZER_MAX_TIME": TIME_LIMIT,
            },
        )

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
            "status": problem.status,
            "solve_time": solve_time,
            "num_iters": num_iters,
            "objective": objective,
        }
    except:
        return {
            "setup_time": None,
            "status": None,
            "solve_time": None,
            "num_iters": None,
            "objective": None,
        }
