import numpy as np
from scipy import sparse
import os
import cvxpy as cp


def write_cvxpy_problem(filename, prob):
    data, _, _ = prob.get_problem_data(cp.CLARABEL)
    p = data["dims"].zero
    l = data["dims"].nonneg
    q = data["dims"].soc
    m = l + sum(q)
    nsoc = len(q)

    c = data["c"]
    try:
        P = data["P"]
        P = sparse.triu(P, format="csc")
    except:
        P = None

    n = len(c)
    A = data["A"][0:p, :]
    b = data["b"][0:p]

    G = data["A"][p::, :]
    h = data["b"][p::]
    write_problem(filename, n, m, p, l, nsoc, P, A, G, c, b, h, q)


## Writes out the problem in the folloiwng binary file
# n:int m:int p:int l:int nsoc:int Pnnz:int Annz:int Gnnz:int
# c:[n doubles] b:[p doubles] h:[m doubles] q:[nsoc doubles]
# P.data:[Pnnz doubles], P.ind:[Pnnz ints], P.ptr:[n + 1 ints]
# A.data:[Annz doubles], A.ind:[Annz ints], A.ptr:[n + 1 ints]
# G.data:[Gnnz doubles], G.ind:[Gnnz ints], G.ptr:[n + 1 ints]
def write_problem(filename, n, m, p, l, nsoc, P, A, G, c, b, h, q):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    P = sparse.triu(P).tocsc() if P else sparse.csc_matrix((n, n))
    A = A.tocsc()
    G = G.tocsc()
    with open(filename, "wb") as f:
        # Header (int32)
        header = np.array([n, m, p, l, nsoc, P.nnz, A.nnz, G.nnz], dtype=np.int32)
        header.tofile(f)

        # Dense vectors
        c.astype(np.float64).tofile(f)
        b.astype(np.float64).tofile(f)
        h.astype(np.float64).tofile(f)
        np.array(q, dtype=np.int32).tofile(f)

        # Helper for CSC
        def dump_csc(M):
            M.data.astype(np.float64).tofile(f)
            M.indices.astype(np.int32).tofile(f)
            M.indptr.astype(np.int32).tofile(f)

        dump_csc(P)
        dump_csc(A)
        dump_csc(G)
