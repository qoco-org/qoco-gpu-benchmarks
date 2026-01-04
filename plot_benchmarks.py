import os, argparse
import pandas as pd
import matplotlib.pyplot as plt

# Enable LaTeX rendering
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
SOLVED_STRINGS = ["QOCO_SOLVED", "SOLVED", "Solved", "optimal"]
PROBLEMS = ["portfolio", "huber", "group_lasso"]


def plot_benchmark(prob_name):
    if not os.path.exists(prob_name):
        print(f"Error: {prob_name} directory does not exist!")
        return

    # Solver names and their display names (using LaTeX formatting)
    solvers = {
        "qoco": r"\textsc{QOCO}",
        "qoco_cuda": r"\textsc{QOCO} (\textsc{GPU})",
        "clarabel": r"\textsc{Clarabel}",
        "cuclarabel": r"\textsc{Clarabel} (\textsc{GPU})",
        "mosek": r"\textsc{MOSEK}",
    }

    # Read all CSV files and plot
    plt.figure(figsize=(10, 6))

    for solver_name, display_name in solvers.items():
        csv_path = os.path.join(prob_name, f"{solver_name}_results.csv")

        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found, skipping...")
            continue

        # Read CSV
        df = pd.read_csv(csv_path)

        # Only plot data where the solver succeeds
        df = df[df["status"].isin(SOLVED_STRINGS)]

        # Filter out rows with None/NaN values
        df = df.dropna(subset=["size", "setup_time", "solve_time"])

        if len(df) == 0:
            print(f"Warning: No valid data in {csv_path}, skipping...")
            continue

        # Calculate runtime
        df["runtime"] = df["setup_time"] + df["solve_time"]

        # Sort by size for better plotting
        df = df.sort_values("size")

        # Plot
        plt.plot(
            df["size"],
            df["runtime"],
            marker="o",
            label=display_name,
            linewidth=2,
            markersize=6,
        )

    plt.xlabel(r"Problem Size", fontsize=12)
    plt.ylabel(r"Runtime (seconds)", fontsize=12)
    plt.title(prob_name, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xscale("log")
    plt.yscale("log")

    # Save plot
    output_path = os.path.join(prob_name, "runtime_vs_size.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {output_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot benchmark results")
    parser.add_argument(
        "--problems",
        required=True,
        type=str,
        help="Problem name (e.g. portfolio, huber) or 'all'",
    )
    args = parser.parse_args()

    if args.problems == "all":
        for prob in PROBLEMS:
            plot_benchmark(prob)
    else:
        if args.problems not in PROBLEMS:
            raise ValueError(
                f"Unknown problem '{args.problems}'. "
                f"Available options: {PROBLEMS + ['all']}"
            )
        plot_benchmark(args.problems)


if __name__ == "__main__":
    main()
