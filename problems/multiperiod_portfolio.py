import numpy as np
import cvxpy as cp
from scipy import sparse


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
    con = [w[:, 0] == w0]
    for t in range(1, T + 1):
        F = sparse.random(n, k, density=0.5, data_rvs=np.random.randn, format="csc")
        D = sparse.diags(np.random.rand(n) * np.sqrt(k), format="csc")
        mu = np.random.randn(n)

        z = np.sqrt(D) @ w[:, t]
        con += [
            cp.sum(w[:, t]) == 1,
            y[:, t] == F.T @ w[:, t],
            0.0 <= y[:, t],
            y[:, t] <= 0.01,
            cp.norm(w[:, t], 1) <= Lmax,
        ]
        obj += (
            cp.sum_squares(z)
            + cp.sum_squares(y[:, t])
            - (1 / gamma) * (mu.T @ w[:, t])
            + cp.sum_squares(w[:, t] - w[:, t - 1])
        )

    prob = cp.Problem(cp.Minimize(obj), con)
    return prob
