import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

PROBLEMS = [
    "group_lasso",
    "huber",
    "portfolio",
    "multiperiod_portfolio",
    "tv_denoising",
]

solvers = {
    "QOCO": "qoco_results.csv",
    "QOCO-GPU": "qoco_cuda_results.csv",
    "CuClarabel": "cuclarabel_results.csv",
    "Mosek": "mosek_results.csv",
    "Gurobi": "gurobi_results.csv",
}

COLOR = {
    "QOCO": "royalblue",
    "QOCO-GPU": "mediumseagreen",
    "CuClarabel": "darkviolet",
    "Mosek": "firebrick",
    "Gurobi": "coral",
}


def load_all_runtimes(tmax):

    t = {s: [] for s in solvers}

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
                    t[solver].append(tmax)
                else:
                    t[solver].append(r)

    return t


def compute_relative_profile(t, xrange=(0, 2.6), n_tau=3600):

    solver_names = list(t.keys())
    n_prob = len(next(iter(t.values())))

    r = {s: np.zeros(n_prob) for s in solver_names}

    for p in range(n_prob):

        min_time = min(t[s][p] for s in solver_names)

        for s in solver_names:
            r[s][p] = t[s][p] / min_time

    tau_vec = np.logspace(xrange[0], xrange[1], n_tau)

    rho = {"tau": tau_vec}

    for s in solver_names:

        rvals = np.array(r[s])
        rho[s] = np.array([np.sum(rvals <= tau) / n_prob for tau in tau_vec])

    df = pd.DataFrame(rho)
    df.to_csv("relative_profile.csv", index=False)

    return df


def compute_absolute_profile(t, xrange=(-3, 3.5), n_tau=3600):

    solver_names = list(t.keys())
    n_prob = len(next(iter(t.values())))

    tau_vec = np.logspace(xrange[0], xrange[1], n_tau)

    rho = {"tau": tau_vec}

    for s in solver_names:

        times = np.array(t[s])
        rho[s] = np.array([np.sum(times <= tau) / n_prob for tau in tau_vec])

    df = pd.DataFrame(rho)
    df.to_csv("absolute_profile.csv", index=False)

    return df


def plot_relative_profile(df):

    plt.figure()

    for col in df.columns[1:]:
        plt.plot(df["tau"], df[col], label=col, color=COLOR[col])

    plt.xscale("log")
    plt.xlabel("Performance ratio", fontsize=14)
    plt.ylabel("Fraction of problems", fontsize=14)
    plt.legend()
    plt.grid(True)

    plt.savefig("benchmark_relative_profile.pdf", bbox_inches="tight")
    plt.close()


def plot_absolute_profile(df):

    plt.figure()

    for col in df.columns[1:]:
        plt.plot(df["tau"], df[col], label=col, color=COLOR[col])

    plt.xscale("log")
    plt.xlabel("Runtime (seconds)", fontsize=14)
    plt.ylabel("Fraction of problems", fontsize=14)
    plt.legend()
    plt.grid(True)

    plt.savefig("benchmark_absolute_profile.pdf", bbox_inches="tight")
    plt.close()


def main():

    tmax = 3600

    # load runtimes
    t = load_all_runtimes(tmax)

    # compute profiles
    df_rel = compute_relative_profile(t)
    df_abs = compute_absolute_profile(t)

    # plot
    plot_relative_profile(df_rel)
    plot_absolute_profile(df_abs)


if __name__ == "__main__":
    main()
