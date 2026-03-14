import os
import numpy as np
import pandas as pd
import math

PROBLEMS = ["group_lasso", "huber", "portfolio", "multiperiod_portfolio", "tv_denoising"]

SOLVED_STRINGS = ["QOCO_SOLVED", "SOLVED", "Solved", "optimal"]

solvers = {
    "QOCO-GPU": "qoco_cuda_results.csv",
    "QOCO": "qoco_results.csv",
    "CuClarabel": "cuclarabel_results.csv",
    "Mosek": "mosek_results.csv",
    "Gurobi": "gurobi_results.csv",
}


def compute_shifted_geometric_mean(tmax=3600):

    t = {s: [] for s in solvers}
    fail = {s: 0 for s in solvers}

    for problem in PROBLEMS:

        for solver, file in solvers.items():

            path = os.path.join(problem, file)
            df = pd.read_csv(path)

            runtime = df["setup_time"] + df["solve_time"]

            if "status" in df.columns:
                status = df["status"]
            else:
                status = ["optimal"] * len(df)

            for r, st in zip(runtime, status):

                if r > tmax or math.isnan(r):
                    fail[solver] += 1
                    t[solver].append(tmax)
                else:
                    t[solver].append(r)

    n_prob = len(next(iter(t.values())))

    # shifted geometric mean (same formula as your code)
    rs = {}
    for s in solvers:
        prod = 1.0
        for p in range(n_prob):
            prod *= (1 + t[s][p])
        rs[s] = prod ** (1 / n_prob) - 1

    # normalize by best solver
    mings = min(rs.values())

    for s in solvers:
        rs[s] /= mings
        fail[s] = 100 * fail[s] / n_prob
    write_latex_table(rs, fail)


def write_latex_table(rs, fail):

    solver_names = list(solvers.keys())

    best_sgm = min(rs.values())
    best_fail = min(fail.values())

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{ \bf Shifted geometric means and failure rates for benchmark problems}")
    lines.append(r"\begin{tabular}{l" + "c"*len(solver_names) + "}")
    lines.append(r"\toprule")

    lines.append(" & " + " & ".join(solver_names) + r" \\")
    lines.append(r"\midrule")

    # Shifted GM row
    row = "Shifted GM"
    for s in solver_names:
        val = "%.2f" % rs[s]
        if abs(rs[s] - best_sgm) < 1e-12:
            val = r"\textbf{" + val + "}"
        row += " & " + val
    row += r" \\"
    lines.append(row)

    # Failure rate row
    row = "Failure Rate (\%)"
    for s in solver_names:
        val = "%.1f" % fail[s]
        if abs(fail[s] - best_fail) < 1e-12:
            val = r"\textbf{" + val + "}"
        row += " & " + val
    row += r" \\"
    lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open("sgm_table.tex", "w") as f:
        f.write("\n".join(lines))

if __name__ == "__main__":
    compute_shifted_geometric_mean()