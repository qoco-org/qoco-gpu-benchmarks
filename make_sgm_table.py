import os
import numpy as np
import pandas as pd
import math

PROBLEMS = ["group_lasso", "huber", "portfolio", "multiperiod_portfolio", "tv_denoising"]

SOLVED_STRINGS = ["QOCO_SOLVED", "SOLVED", "Solved", "optimal"]

solvers = {
    "CuClarabel": "cuclarabel_results.csv",
    "Gurobi": "gurobi_results.csv",
    "Mosek": "mosek_results.csv",
    "QOCO": "qoco_results.csv",
    "QOCO-GPU": "qoco_cuda_results.csv",
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
    breakpoint()
    for s in solvers:
        prod = 1.0
        for p in range(n_prob):
            prod *= (1 + t[s][p])
            if s == "Mosek":
                print("Prod: ", prod)
                print("factor: ", 1 + t[s][p])

        rs[s] = prod ** (1 / n_prob) - 1

    # normalize by best solver
    mings = min(rs.values())

    for s in solvers:
        rs[s] /= mings
        fail[s] = 100 * fail[s] / n_prob
    write_latex_table(rs, fail)


def write_latex_table(rs, fail):

    solver_names = list(solvers.keys())

    lines = []
    lines.append(r"\begin{tabular}{l" + "c"*len(solver_names) + "}")
    lines.append(r"\toprule")

    lines.append(" & " + " & ".join(solver_names) + r" \\")
    lines.append(r"\midrule")

    row = "Shifted GM"
    for s in solver_names:
        row += " & %.2f" % rs[s]
    row += r" \\"
    lines.append(row)

    row = "Failure Rate (\%)"
    for s in solver_names:
        row += " & %.1f" % fail[s]
    row += r" \\"
    lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    with open("sgm_table.tex", "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    compute_shifted_geometric_mean()