from problems.portfolio import portfolio

k = 10
prob = portfolio(k)
# prob.solve(verbose=True, solver="CUCLARABEL")
# cuclarabel_setup_time = 0 if prob.solver_stats.setup_time is None else prob.solver_stats.setup_time
# cuclarabel_solve_time = prob.solver_stats.solve_time

prob.solve(verbose=True, solver="QOCO", algebra="cuda")
prob.solve(verbose=True, solver="QOCO", algebra="cuda")
qoco_gpu_setup_time = 0 if prob.solver_stats.setup_time is None else prob.solver_stats.setup_time
qoco_gpu_solve_time = prob.solver_stats.solve_time

# print("CUCLARABEL Setup Time: " + str(cuclarabel_setup_time))
# print("CUCLARABEL Solve Time: " + str(cuclarabel_solve_time))
print("QOCO GPU Setup Time: " + str(qoco_gpu_setup_time))
print("QOCO GPU Solve Time: " + str(qoco_gpu_solve_time))

