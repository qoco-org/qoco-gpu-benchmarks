python run_benchmarks.py --problems all

mkdir -p figures
python make_benchmark_table.py
python make_sgm_table.py
python plot_performance_profiles.py