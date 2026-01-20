import argparse
from pathlib import Path
from typing import Any
import os
from solvers import SOLVERS, get_problem_size
from convert import mps_to_standard_form
from utils import write_results

TOO_BIG = ["in.mps"]

SKIP = ["supportcase2.mps", "tpl-tub-ss16.mps", "neos-3634244-kauru.mps", "neos-4321076-ruwer.mps", "neos-5106984-jizera.mps", "rail02.mps", "neos-5223573-tarwin.mps", "physiciansched3-4.mps", "tpl-tub-ws1617.mps", "kosova1.mps", "highschool1-aigio.mps", "ns1644855.mps", "neos-5041822-cockle.mps", "neos-5138690-middle.mps", "neos-5116085-kenana.mps", "neos-5114902-kasavu.mps", "neos-3025225-shelon.mps", "neos-5118851-kowhai.mps", "neos-4966258-blicks.mps", "neos-5118834-korana.mps", "academictimetablebig.mps", "neos-4972461-bolong.mps", "ex10.mps", "woodlands09.mps", "graph40-80-1rand.mps", "sing17.mps", "satellites4-25.mps", "in.mps", "bab3.mps", "shs1042.mps", "bab2.mps", "neos-5049753-cuanza.mps", "shs1014.mps", "s250r10.mps", "square37.mps", "ivu06.mps", "stp3d.mps", "neos-5108386-kalang.mps", "neos-5251015-ogosta.mps", "fhnw-binschedule1.mps", "rwth-timetable.mps", "physiciansched3-3.mps", "neos-5273874-yomtsa.mps", "neos-5123665-limmat.mps", "supportcase19.mps", "neos-3322547-alsek.mps", "supportcase43.mps", "neos-5104907-jarama.mps", "s100.mps", "supportcase7.mps", "neos-4976951-bunnoo.mps", "neos-4972437-bojana.mps"]

PATH = {
    "mip-relaxations"
}

# @profile
def run_benchmarks(prob_name):
    directory = Path("data/" + prob_name)
    os.makedirs(prob_name, exist_ok=True)

    # Dict for results
    results = {solver_name: [] for solver_name in SOLVERS.keys()}

    # Track which CUDA solvers have been warmed up
    cuda_warmed_up = set[Any]()
    cuda_solvers = {"qoco_cuda", "cuclarabel"}

    # Run benchmarks for each n value
    for file_path in directory.iterdir():
        problem_name = file_path.stem
        if problem_name in SKIP:
            continue
        print(problem_name)
        data = mps_to_standard_form(file_path._raw_path)
        problem_size = get_problem_size(data)
        for solver_name, solver_func in SOLVERS.items():
            print(f"  Running {solver_name}...")

            # Warmup CUDA solvers on first call
            if solver_name in cuda_solvers and solver_name not in cuda_warmed_up:
                print(f"    Warming up {solver_name} (CUDA initialization)...")
                _ = solver_func(data)
                cuda_warmed_up.add(solver_name)
            if solver_name == "mosek":
                stats = solver_func(data)
            else:
                stats = solver_func(data)
            stats["size"] = problem_size
            stats["name"] = file_path.stem
            results[solver_name].append(stats)

    write_results(results, prob_name)


def main():
    parser = argparse.ArgumentParser(description="Run benchmark problems")
    parser.add_argument(
        "--problems",
        required=True,
        type=str,
        help="Problem name (e.g. mip-relaxations) or 'all'",
    )
    args = parser.parse_args()
    run_benchmarks(args.problems)


if __name__ == "__main__":
    main()
