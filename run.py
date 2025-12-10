from problems.portfolio import *
from solvers import run_qoco

np.random.seed(0)
k = 10

data = portfolio_handparsed(k)
res = run_qoco(data, algebra="cuda")
print(res)
