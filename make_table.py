import pandas as pd

NAME = "tv_denoising"

solvers = {
    "CuClarabel": NAME + "/cuclarabel_results.csv",
    "Gurobi": NAME + "/gurobi_results.csv",
    "Mosek": NAME + "/mosek_results.csv",
    "QOCO": NAME + "/qoco_results.csv",
    "QOCO-GPU": NAME + "/qoco_cuda_results.csv",
}

def latex_escape(s):
    return (str(s)
            .replace("\\", r"\textbackslash ")
            .replace("_", r"\_")
            .replace("&", r"\&")
            .replace("%", r"\%")
            .replace("#", r"\#"))

dfs = {}

# read CSVs and compute runtime
for solver, path in solvers.items():
    df = pd.read_csv(path)
    df["runtime"] = df["setup_time"] + df["solve_time"]
    dfs[solver] = df[["name", "size", "runtime"]]

# merge all solver tables
merged = None
for solver, df in dfs.items():
    df = df.rename(columns={"runtime": solver})
    if merged is None:
        merged = df
    else:
        merged = merged.merge(df[["name", solver]], on="name", how="outer")

# keep size column from the first dataframe
merged["size"] = merged["size"].ffill()

# sort by size (optional but usually nicer)
merged = merged.sort_values("size")

solver_names = list(solvers.keys())

lines = []

lines.append(r"\begin{table}[ht]")
lines.append(r"\begin{tabular}{l r " + " ".join(["r"]*len(solver_names)) + "}")
lines.append(r"\toprule")
lines.append("Problem & Size & " + " & ".join(solver_names) + r" \\")
lines.append(r"\midrule")

for _, row in merged.iterrows():

    runtimes = [row[s] for s in solver_names if not pd.isna(row[s])]
    best = min(runtimes) if runtimes else None

    cells = []
    for s in solver_names:
        val = row[s]

        if pd.isna(val):
            cells.append("-")
        elif best is not None and val == best:
            cells.append(r"\winner %.3f" % val)
        else:
            cells.append("%.3f" % val)

    name = latex_escape(row["name"])

    line = (
        f"{name} & {int(row['size'])} & "
        + " & ".join(cells)
        + r" \\"
    )

    lines.append(line)

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\captionsetup{labelfont=bf}")
lines.append(rf"\caption{{ \bf {latex_escape(NAME).capitalize()} runtime in seconds.}}")
lines.append(rf"\label{{tab:{NAME}_benchmarks}}")
# lines.append("")
lines.append(r"\end{table}")

with open(f"{NAME}_table.tex", "w") as f:
    f.write("\n".join(lines))