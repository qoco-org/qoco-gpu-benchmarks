import gurobipy as gp
import numpy as np
from scipy.sparse import csc_matrix
from solvers import ProblemData


def mps_to_standard_form(mps_file):
    """
    Reads an MPS (.mps or .mps.gz) with Gurobi and returns

        min c^T x
        s.t. A x = b
             G x <= h

    where A and G are scipy.sparse.csc_matrix.
    """

    m = gp.read(mps_file)
    m.update()

    vars = m.getVars()
    constrs = m.getConstrs()

    n = len(vars)

    # ----------------------
    # Objective
    # ----------------------
    c = np.array([v.Obj for v in vars])

    # ----------------------
    # Sparse assembly buffers
    # ----------------------
    A_rows, A_cols, A_data = [], [], []
    b = []

    G_rows, G_cols, G_data = [], [], []
    h = []

    eq_row = 0
    ineq_row = 0

    # ----------------------
    # Constraints
    # ----------------------
    for con in constrs:
        expr = m.getRow(con)
        rhs = con.RHS
        sense = con.Sense

        if sense == gp.GRB.EQUAL:
            for k in range(expr.size()):
                v = expr.getVar(k)
                A_rows.append(eq_row)
                A_cols.append(v.index)
                A_data.append(expr.getCoeff(k))
            b.append(rhs)
            eq_row += 1

        elif sense == gp.GRB.LESS_EQUAL:
            for k in range(expr.size()):
                v = expr.getVar(k)
                G_rows.append(ineq_row)
                G_cols.append(v.index)
                G_data.append(expr.getCoeff(k))
            h.append(rhs)
            ineq_row += 1

        elif sense == gp.GRB.GREATER_EQUAL:
            for k in range(expr.size()):
                v = expr.getVar(k)
                G_rows.append(ineq_row)
                G_cols.append(v.index)
                G_data.append(-expr.getCoeff(k))
            h.append(-rhs)
            ineq_row += 1

    # ----------------------
    # Variable bounds → Gx ≤ h
    # ----------------------
    for j, v in enumerate(vars):
        if v.UB < gp.GRB.INFINITY:
            G_rows.append(ineq_row)
            G_cols.append(j)
            G_data.append(1.0)
            h.append(v.UB)
            ineq_row += 1

        if v.LB > -gp.GRB.INFINITY:
            G_rows.append(ineq_row)
            G_cols.append(j)
            G_data.append(-1.0)
            h.append(-v.LB)
            ineq_row += 1

    # ----------------------
    # Build sparse matrices
    # ----------------------
    A = csc_matrix(
        (A_data, (A_rows, A_cols)),
        shape=(eq_row, n),
    )

    G = csc_matrix(
        (G_data, (G_rows, G_cols)),
        shape=(ineq_row, n),
    )

    b = np.array(b)
    h = np.array(h)
    return ProblemData(
        n=n,
        m=ineq_row,
        p=eq_row,
        P=None,
        c=c,
        A=A,
        b=b,
        G=G,
        h=h,
        l=ineq_row,
        nsoc=0,
        q=[],
    )
