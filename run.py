from problems.portfolio import *
from solvers import *

np.random.seed(0)
k = 3000

data = portfolio_handparsed(k)
# res = run_qoco(data, algebra="cuda")
res = run_clarabel(data, algebra="cuda")

print(res)
