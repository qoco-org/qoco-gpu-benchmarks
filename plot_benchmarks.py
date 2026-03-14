import os
import pandas as pd
import matplotlib.pyplot as plt

# Enable LaTeX
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"

COLOR = {
    "qoco": "royalblue",
    "qoco_cuda": "mediumseagreen",
    "cuclarabel": "darkviolet",
    "mosek": "firebrick",
    "gurobi": "coral",
}

SOLVED_STRINGS = ["QOCO_SOLVED", "SOLVED", "Solved", "optimal"]

PROBLEMS = [
    "portfolio",
    "huber",
    "group_lasso",
    "multiperiod_portfolio",
    "tv_denoising",
]

SOLVERS = {
    "qoco": r"\textsc{QOCO}",
    "qoco_cuda": r"\textsc{QOCO-GPU}",
    "cuclarabel": r"\textsc{CuClarabel}",
    "mosek": r"\textsc{Mosek}",
    "gurobi": r"\textsc{Gurobi}",
}


def plot_problem(ax, prob_name):
    """Plot one benchmark problem on an axis."""

    if not os.path.exists(prob_name):
        print(f"Warning: {prob_name} directory not found")
        return

    for solver_name, display_name in SOLVERS.items():

        csv_path = os.path.join(prob_name, f"{solver_name}_results.csv")

        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found")
            continue

        df = pd.read_csv(csv_path)

        df = df.dropna(subset=["size", "setup_time", "solve_time"])

        if len(df) == 0:
            continue

        df["runtime"] = df["setup_time"] + df["solve_time"]
        df = df[df["runtime"] <= 3600.0]

        df = df.sort_values("size")

        ax.plot(
            df["size"],
            df["runtime"],
            "o-",
            linewidth=2,
            markersize=5,
            label=display_name,
            color=COLOR[solver_name]
        )

    ax.set_xscale("log")
    ax.set_yscale("log")

    # ax.set_xlabel("Problem Size")
    # ax.set_ylabel("Runtime (seconds)")

    ax.set_title(prob_name.replace("_", " ").title(), usetex=True)

    ax.grid(True, alpha=0.3)


def main():

    plt.figure(figsize=(8.5, 11))

    axes = []

    # create subplots
    for i, prob in enumerate(PROBLEMS):
        ax = plt.subplot(3, 2, i + 1)
        axes.append(ax)
        plot_problem(ax, prob)


    # center the fifth plot
    ax5 = axes[4]
    pos = ax5.get_position()

    xright = axes[3].get_position().x0

    ax5.set_position([
        0.5 * (pos.x0 + xright),
        pos.y0,
        pos.width,
        pos.height,
    ])

    # legend
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles, labels, loc="center right", bbox_to_anchor=(1.5, 0.5))

    plt.savefig(
        "benchmark_runtime.pdf",
        dpi=300,
        bbox_inches="tight",
    )

if __name__ == "__main__":
    main()