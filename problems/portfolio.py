import numpy as np
import cvxpy as cp
from scipy import sparse
from solvers import ProblemData


def portfolio_cvxpy(k):
    np.random.seed(42)
    n = 100 * k
    F = sparse.random(n, k, density=0.5, data_rvs=np.random.randn, format="csc")
    D = sparse.diags(np.random.rand(n) * np.sqrt(k), format="csc")
    mu = np.random.randn(n)

    gamma = 1

    x = cp.Variable(n)
    y = cp.Variable(k)

    obj = cp.Minimize(
        cp.quad_form(x, D) + cp.quad_form(y, sparse.eye(k)) - (1 / gamma) * (mu.T @ x)
    )
    con = [cp.sum(x) == 1, F.T @ x == y, 0 <= x]
    prob = cp.Problem(obj, con)
    return prob


def portfolio_handparsed(k):
    np.random.seed(42)
    n = 100 * k
    F = sparse.random(n, k, density=0.5, data_rvs=np.random.randn, format="csc")
    D = sparse.diags(np.random.rand(n) * np.sqrt(k), format="csc")
    mu = np.random.randn(n)

    gamma = 1

    # Define c as -1/gamma * [mu;0]
    c = -1 / gamma * np.concatenate([mu, np.zeros(k)])

    # Define P as [D 0; 0 I]
    P = 2 * sparse.block_diag([D, sparse.eye(k)])

    # Define A as [F^T -I; 1^T 0]
    A = sparse.bmat(
        [[F.T, -sparse.eye(k)], [np.ones((1, n)), sparse.csr_matrix((1, k))]],
        format="csc",
    )

    # Define G as [-I 0]
    G = sparse.bmat([[-sparse.eye(n), sparse.csr_matrix((n, k))]], format="csc")

    # Define b as [0; 1]
    b = np.concatenate([np.zeros(k), np.ones(1)])

    # Define h as [0; 0; 0]
    h = np.zeros(n)

    return ProblemData(
        n=n + k, m=n, p=k + 1, P=P, c=c, A=A, b=b, G=G, h=h, l=n, nsoc=0, q=[]
    )
