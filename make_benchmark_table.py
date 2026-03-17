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
    "Gurobi": "gurobi_results.csv",
}


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

        # Compute setup fraction (only meaningful for QOCO-GPU)
        if solver == "QOCO-GPU":
            df["setup_frac"] = df["setup_time"] / df["runtime"]
        else:
            df["setup_frac"] = pd.NA

        # Mark timeouts
        df.loc[
            (df["runtime"] > 3600.0) | (~df["status"].isin(SOLVED_STRINGS)),
            "runtime"
        ] = pd.NA
        df.loc[df["runtime"].isna(), "setup_frac"] = pd.NA

        dfs[solver] = df[["name", "size", "runtime", "setup_frac"]]

    merged = None
    for solver, df in dfs.items():
        df = df.rename(
            columns={"runtime": solver, "setup_frac": f"{solver}_setup_frac"}
        )

        if merged is None:
            merged = df
        else:
            merged = merged.merge(
                df[["name", solver, f"{solver}_setup_frac"]], on="name", how="outer"
            )

    merged["size"] = merged["size"].ffill()
    merged = merged.sort_values("size")
    merged["problem_group"] = problem_name

    return merged


def make_benchmark_table():
    tables = [load_problem(p) for p in PROBLEMS]
    merged = pd.concat(tables, ignore_index=True)

    solver_names = list(solvers.keys())

    lines = []

    lines.append(r"{\footnotesize")
    lines.append(r"\begin{longtable}{l r " + " ".join(["r"] * len(solver_names)) + "}")
    lines.append(
        r"\caption{\bf Runtime in seconds for benchmark problems (QOCO-GPU shows setup time percentage in parentheses)}"
    )
    lines.append(r"\label{tab:solver_benchmarks} \\")
    lines.append("")
    lines.append(r"\toprule")
    lines.append("Problem & Size & " + " & ".join(solver_names) + r" \\")
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append("")
    lines.append(r"\toprule")
    lines.append("Problem & Size & " + " & ".join(solver_names) + r" \\")
    lines.append(r"\midrule")
    lines.append(r"\endhead")
    lines.append("")
    lines.append(r"\midrule")
    lines.append(
        r"\multicolumn{"
        + str(len(solver_names) + 2)
        + r"}{r}{\footnotesize Continued on next page} \\"
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

        # Determine best runtime (ignore NaNs)
        runtimes = [row[s] for s in solver_names if not pd.isna(row[s])]
        best = min(runtimes) if runtimes else None

        cells = []
        for s in solver_names:
            val = row[s]

            if pd.isna(val):
                cells.append("-")
                continue

            frac = row.get(f"{s}_setup_frac", pd.NA)

            # Format QOCO-GPU with optional setup %
            if s == "QOCO-GPU":
                if pd.isna(frac) or frac < 1e-6:
                    base = f"{val:.3f}"
                else:
                    pct = int(round(100 * frac))
                    base = f"{val:.3f} ({pct}\\%)"
            else:
                base = f"{val:.3f}"

            # Highlight winner (based on runtime only)
            if best is not None and val == best:
                if s == "QOCO-GPU" and not pd.isna(frac) and frac >= 1e-6:
                    pct = int(round(100 * frac))
                    base = f"\\winner {val:.3f} ({pct}\\%)"
                else:
                    base = f"\\winner {val:.3f}"

            cells.append(base)

        name = latex_escape(row["name"])

        line = f"{name} & {int(row['size'])} & " + " & ".join(cells) + r" \\"

        lines.append(line)

    lines.append(r"\end{longtable}")
    lines.append(r"}")

    with open("benchmark_table.tex", "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    make_benchmark_table()
