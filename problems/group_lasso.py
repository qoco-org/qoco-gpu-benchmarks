import cvxpy as cp
import numpy as np
from scipy import sparse
from solvers import ProblemData


def group_lasso_cvxpy(ngroups):
    np.random.seed(42)
    group_size = 10
    n = ngroups * group_size
    m = 25 * n
    lam = 1

    xtrue = np.zeros(n)
    if ngroups > 1:
        for i in range(int(ngroups / 2)):
            xtrue[i * group_size : (i + 1) * group_size] = np.random.randn(group_size)
    else:
        xtrue[0:group_size] = np.random.randn(group_size)

    A = sparse.random(m, n, density=0.1, data_rvs=np.random.randn, format="csc")
    e = np.random.randn(m) / n
    b = A @ xtrue + e
    y = cp.Variable(m)
    x = cp.Variable(n)
    con = [y == A @ x - b]

    x_reshaped = cp.reshape(x, (ngroups, group_size), order="C")
    obj = cp.sum_squares(y) + lam * cp.mixed_norm(x_reshaped, p=2, q=1)
    prob = cp.Problem(cp.Minimize(obj), con)
    return prob


def group_lasso_handparsed(ngroups):
    np.random.seed(42)
    group_size = 10
    n = ngroups * group_size
    m = 25 * n
    lam = 1

    xtrue = np.zeros(n)
    if ngroups > 1:
        for i in range(int(ngroups / 2)):
            xtrue[i * group_size : (i + 1) * group_size] = np.random.randn(group_size)
    else:
        xtrue[0:group_size] = np.random.randn(group_size)

    A = sparse.random(m, n, density=0.1, data_rvs=np.random.randn, format="csc")
    e = np.random.randn(m) / n
    b = A @ xtrue + e

    Im = sparse.identity(m, format="csc")
    P = sparse.block_diag(
        [
            sparse.csc_matrix((n, n)),
            2 * Im,
            sparse.csc_matrix((ngroups, ngroups)),
        ],
        format="csc",
    )
    c = np.concatenate([np.zeros(n), np.zeros(m), lam * np.ones(ngroups)])
    Zng = sparse.csc_matrix((m, ngroups))
    A = sparse.hstack([A, -Im, Zng], format="csc")

    Grows = (group_size + 1) * ngroups
    Gx = np.kron(
        np.eye(ngroups), np.vstack([np.zeros((1, group_size)), np.eye(group_size)])
    )
    Gy = sparse.csc_matrix((Grows, m))

    Gtblock = np.zeros((group_size + 1, 1))
    Gtblock[0][0] = 1
    Gt = np.kron(np.eye(ngroups), Gtblock)
    G = -sparse.hstack([Gx, Gy, Gt], format="csc")
    h = np.zeros(Grows)
    q = (group_size + 1) * np.ones(ngroups, dtype=np.int32)

    return ProblemData(
        n=n + m + ngroups,
        m=Grows,
        p=m,
        P=P,
        c=c,
        A=A,
        b=b,
        G=G,
        h=h,
        l=0,
        nsoc=ngroups,
        q=q,
    )
