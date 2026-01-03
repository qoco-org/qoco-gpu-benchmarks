import os
import pandas as pd
import matplotlib.pyplot as plt

# Enable LaTeX rendering
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
SOLVED_STRINGS = ["QOCO_SOLVED", "SOLVED", "Solved", "optimal"]


def plot_suitesparse_results():
    """
    Plot runtime (setup_time + solve_time) vs size for all solvers.
    """
    suitesparse_dir = "suitesparse"

    if not os.path.exists(suitesparse_dir):
        print(f"Error: {suitesparse_dir} directory does not exist!")
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
        csv_path = os.path.join(suitesparse_dir, f"{solver_name}_results.csv")

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
        plt.scatter(
            df["size"],
            df["runtime"],
            marker="o",
            label=display_name,
            linewidth=2,
        )

    plt.xlabel(r"Problem Size $\mathrm{nnz}(A) + \mathrm{nnz}(P)$", fontsize=12)
    plt.ylabel(r"Runtime (seconds)", fontsize=12)
    plt.title(r"SuiteSparse: Runtime vs Problem Size", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xscale("log")
    plt.yscale("log")

    # Save plot
    output_path = os.path.join(suitesparse_dir, "runtime_vs_size.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {output_path}")

    plt.show()


if __name__ == "__main__":
    plot_suitesparse_results()
