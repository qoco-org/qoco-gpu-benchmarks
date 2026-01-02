import pandas as pd
import os

def write_results(results):
    for solver_name, solver_results in results.items():
        csv_filename = os.path.join("portfolio", f"{solver_name}_results.csv")
        print(f"Writing {csv_filename}...")
        df = pd.DataFrame(solver_results)
        columns = [
            "size",
            "status",
            "setup_time",
            "solve_time",
            "num_iters",
            "objective",
        ]
        columns = [col for col in columns if col in df.columns]
        df[columns].to_csv(csv_filename, index=False)
        print(f"  Wrote {len(solver_results)} rows to {csv_filename}")
