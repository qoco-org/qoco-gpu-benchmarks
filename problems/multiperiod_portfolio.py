import numpy as np
import cvxpy as cp
from scipy import sparse
from solvers import ProblemData


def multiperiod_portfolio_cvxpy(T):
    np.random.seed(42)
    k = 50
    n = 5000
    Lmax = 1.6

    gamma = 1

    w = cp.Variable((n, T + 1))
    y = cp.Variable((k, T + 1))

    w0 = np.random.rand(n)
    w0 = w0 / np.sum(w0)

    obj = 0.0
    con = [w[:,0] == w0]
    for t in range(1, T+1):
        F = sparse.random(n, k, density=0.5, data_rvs=np.random.randn, format="csc")
        D = sparse.diags(np.random.rand(n) * np.sqrt(k), format="csc")
        mu = np.random.randn(n)

        z = np.sqrt(D) @ w[:,t]
        con += [cp.sum(w[:,t]) == 1, y[:,t] == F.T @ w[:,t], 0.0 <= y[:, t], y[:,t] <= 0.01, cp.norm(w[:,t], 1) <= Lmax]
        obj += cp.sum_squares(z) + cp.sum_squares(y[:, t]) - (1 / gamma) * (mu.T @ w[:, t]) + cp.sum_squares(w[:,t] - w[:,t-1])

    prob = cp.Problem(cp.Minimize(obj), con)
    return prob


# def multiperiod_portfolio_handparsed(k):
    # np.random.seed(42)
    # n = 100 * k
    # F = sparse.random(n, k, density=0.5, data_rvs=np.random.randn, format="csc")
    # D = sparse.diags(np.random.rand(n) * np.sqrt(k), format="csc")
    # mu = np.random.randn(n)

    # gamma = 1

    # # Define c as -1/gamma * [mu;0]
    # c = -1 / gamma * np.concatenate([mu, np.zeros(k)])

    # # Define P as [D 0; 0 I]
    # P = 2 * sparse.block_diag([D, sparse.eye(k)])

    # # Define A as [F^T -I; 1^T 0]
    # A = sparse.bmat(
    #     [[F.T, -sparse.eye(k)], [np.ones((1, n)), sparse.csr_matrix((1, k))]],
    #     format="csc",
    # )

    # # Define G as [-I 0]
    # G = sparse.bmat([[-sparse.eye(n), sparse.csr_matrix((n, k))]], format="csc")

    # # Define b as [0; 1]
    # b = np.concatenate([np.zeros(k), np.ones(1)])

    # # Define h as [0; 0; 0]
    # h = np.zeros(n)

    # return ProblemData(
    #     n=n + k, m=n, p=k + 1, P=P, c=c, A=A, b=b, G=G, h=h, l=n, nsoc=0, q=[]
    # )
