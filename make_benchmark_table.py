import pandas as pd

PROBLEMS = [
    "huber",
    "portfolio",
    "multiperiod_portfolio",
    "group_lasso",
    "tv_denoising",
]

SOLVED_STRINGS = ["QOCO_SOLVED", "SOLVED", "Solved", "optimal"]

solvers = {
    "QOCO-GPU": "qoco_cuda_results.csv",
    "QOCO": "qoco_results.csv",
    "CuClarabel": "cuclarabel_results.csv",
    "Mosek": "mosek_results.csv",
}

# Solvers shown as a single total-runtime column (in table order).
SINGLE_SOLVERS = ["QOCO", "CuClarabel", "Mosek"]

# tv_denoising instances are named by suffix index; map each index to the
# scikit-image dataset used (see problems/tv_denoising.py).
TV_DENOISING_NAMES = [
    "chelsea",
    "astronaut",
    "coffee",
    "immunohistochemistry",
    "logo",
    "brick",
    "camera",
    "grass",
]


def format_size(n):
    """Group digits of an integer in threes with LaTeX thin spaces, e.g.
    100600000 -> 100\\,600\\,000."""
    return f"{int(n):,}".replace(",", r"\,")


def latex_escape(s):
    return (
        str(s)
        .replace("\\", r"\textbackslash ")
        .replace("_", r"\_")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("#", r"\#")
    )


def load_problem(problem_name):
    dfs = {}

    for solver, file in solvers.items():
        path = f"{problem_name}/{file}"
        df = pd.read_csv(path)

        df["runtime"] = df["setup_time"] + df["solve_time"]

        # Analysis time is only reported by QOCO-GPU.
        if solver == "QOCO-GPU":
            df["analysis_time"] = df["analysis_time"]
        else:
            df["analysis_time"] = pd.NA

        # Mark timeouts / failures as missing.
        failed = (df["runtime"] > 3600.0) | (~df["status"].isin(SOLVED_STRINGS))
        df.loc[failed, ["runtime", "setup_time", "solve_time", "analysis_time"]] = pd.NA

        dfs[solver] = df[
            ["name", "size", "runtime", "setup_time", "solve_time", "analysis_time"]
        ]

    merged = None
    for solver, df in dfs.items():
        df = df.rename(
            columns={
                "runtime": solver,
                "setup_time": f"{solver}_setup",
                "solve_time": f"{solver}_solve",
                "analysis_time": f"{solver}_analysis",
            }
        )

        cols = ["name", solver, f"{solver}_setup", f"{solver}_solve", f"{solver}_analysis"]

        if merged is None:
            merged = df[["size"] + cols]
        else:
            merged = merged.merge(df[cols], on="name", how="outer")

    merged["size"] = merged["size"].ffill()

    # tv_denoising's _0.._N suffixes are out of order. Sort by the suffix so the
    # indices increase monotonically, keeping each instance's size and data
    # attached to its name. Other families are sorted ascending by size.
    if problem_name == "tv_denoising":
        suffix = merged["name"].str.rsplit("_", n=1).str[-1].astype(int)
        merged = merged.assign(_suffix=suffix).sort_values("_suffix")
        merged = merged.reset_index(drop=True)
        # Replace the suffix index with the dataset name it corresponds to.
        merged["name"] = merged["_suffix"].map(
            lambda i: f"{problem_name}_{TV_DENOISING_NAMES[i]}"
        )
        merged = merged.drop(columns="_suffix")
    else:
        merged = merged.sort_values("size").reset_index(drop=True)

    merged["problem_group"] = problem_name

    return merged


def make_benchmark_table():
    tables = [load_problem(p) for p in PROBLEMS]
    merged = pd.concat(tables, ignore_index=True)

    # All solvers that contribute a total runtime, used to find the winner.
    total_solvers = list(solvers.keys())

    lines = []

    lines.append(r"{\footnotesize")
    lines.append(r"\begin{longtable}{l r *{3}{r} r r r}")
    lines.append(
        r"\caption{\bf Runtime in seconds for benchmark problems. The QOCO-GPU "
        r"columns split its total runtime into setup and solve time. The "
        r"value in parentheses is the percentage of setup time spent in "
        r"cuDSS's analysis (reordering) phase, the dominant component of "
        r"setup. The fastest "
        r"total runtime for each instance is highlighted.}"
    )
    lines.append(r"\label{tab:solver_benchmarks} \\")
    lines.append("")

    header_group = (
        r" & & \multicolumn{3}{c}{QOCO-GPU} & QOCO & CuClarabel & Mosek \\"
    )
    header_cmid = r"\cmidrule(lr){3-5}"
    header_cols = (
        r"Problem & Size & Setup & Solve & Total & & & \\"
    )

    lines.append(r"\toprule")
    lines.append(header_group)
    lines.append(header_cmid)
    lines.append(header_cols)
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append("")
    lines.append(r"\toprule")
    lines.append(header_group)
    lines.append(header_cmid)
    lines.append(header_cols)
    lines.append(r"\midrule")
    lines.append(r"\endhead")
    lines.append("")
    lines.append(r"\midrule")
    lines.append(
        r"\multicolumn{8}{r}{\footnotesize Continued on next page} \\"
    )
    lines.append(r"\endfoot")
    lines.append("")
    lines.append(r"\bottomrule")
    lines.append(r"\endlastfoot")

    current_group = None

    for _, row in merged.iterrows():

        if current_group is not None and row["problem_group"] != current_group:
            lines.append(r"\midrule")

        current_group = row["problem_group"]

        # Determine best total runtime across all solvers (ignore NaNs).
        totals = [row[s] for s in total_solvers if not pd.isna(row[s])]
        best = min(totals) if totals else None

        cells = []

        # --- QOCO-GPU: setup (analysis), solve, total ---
        gpu_total = row["QOCO-GPU"]
        if pd.isna(gpu_total):
            cells.extend(["--", "--", "--"])
        else:
            setup = row["QOCO-GPU_setup"]
            solve = row["QOCO-GPU_solve"]
            analysis = row["QOCO-GPU_analysis"]

            if pd.isna(analysis) or setup == 0:
                setup_cell = f"{setup:.3f}"
            else:
                pct = int(round(100 * analysis / setup))
                setup_cell = f"{setup:.3f} ({pct}\\%)"

            solve_cell = f"{solve:.3f}"

            if best is not None and gpu_total == best:
                total_cell = f"\\winner {gpu_total:.3f}"
            else:
                total_cell = f"{gpu_total:.3f}"

            cells.extend([setup_cell, solve_cell, total_cell])

        # --- Single-column solvers: total runtime ---
        for s in SINGLE_SOLVERS:
            val = row[s]
            if pd.isna(val):
                cells.append("--")
                continue

            if best is not None and val == best:
                cells.append(f"\\winner {val:.3f}")
            else:
                cells.append(f"{val:.3f}")

        name = latex_escape(row["name"])

        line = f"{name} & {format_size(row['size'])} & " + " & ".join(cells) + r" \\"

        lines.append(line)

    lines.append(r"\end{longtable}")
    lines.append(r"}")

    with open("figures/benchmark_table.tex", "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    make_benchmark_table()
