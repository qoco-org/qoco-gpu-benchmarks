import numpy as np
import cvxpy as cp
from scipy import sparse
from solvers import ProblemData


def huber_cvxpy(n):
    np.random.seed(42)
    m = 10 * n
    A = sparse.random(m, n, density=0.1, data_rvs=np.random.randn, format="csc")
    b = np.random.rand(m)

    x = cp.Variable(n)

    obj = cp.Minimize(cp.sum(cp.huber(A @ x - b)))
    prob = cp.Problem(obj, [])
    return prob


def huber_handparsed(n):
    np.random.seed(42)
    m = 10 * n
    A = sparse.random(m, n, density=0.1, data_rvs=np.random.randn, format="csc")
    b = np.random.rand(m)

    P = sparse.block_diag(
        [
            sparse.csc_matrix((n, n)),
            2 * sparse.identity(m, format="csc"),
            sparse.csc_matrix((m, m)),
            sparse.csc_matrix((m, m)),
        ],
        format="csc",
    )
    c = np.concatenate([np.zeros(n), np.zeros(m), 2 * np.ones(m), 2 * np.ones(m)])

    I = sparse.identity(m, format="csc")
    A = sparse.hstack([A.tocsc(), -I, -I, I], format="csc")

    Zn = sparse.csc_matrix((m, n))
    Zm = sparse.csc_matrix((m, m))

    # First block row: [0  0  -I  0]
    row1 = sparse.hstack([Zn, Zm, -I, Zm], format="csc")

    # Second block row: [0  0   0 -I]
    row2 = sparse.hstack([Zn, Zm, Zm, -I], format="csc")

    G = sparse.vstack([row1, row2], format="csc")
    h = np.zeros(2 * m)

    return ProblemData(
        n=n + 3 * m, m=2 * m, p=m, P=P, c=c, A=A, b=b, G=G, h=h, l=2 * m, nsoc=0, q=[]
    )
