from linear import PropagateSatelliteLinearMan
from second_order import PropagateSatelliteSecondMan
import numpy as np

def report_rmse(errors):

    errors = np.asarray(errors)                    # (N, 10) rows from _error_row

    rmse = np.sqrt(np.mean(errors**2, axis=0))     # RMSE per component, over runs
    cols  = ["x", "y", "z", "vx", "vy", "vz", "dvx", "dvy", "dvz", "t1"]

    print(f"RMSE over {len(errors)} runs")
    print(f"{'Position, km':^32}{'Velocity, m/s':^32}{'Impulsive vector, m/s':^32}{'Epoch, s':^10}")
    print("".join(f"{c:>10}" for c in cols))
    print("".join(f"{v:10.4f}" for v in rmse))

    return rmse

def run_monte_carlo(type_order, case = "std", n_runs = 100, num_measurements = 180, time_split = 10, print_every = 10):

    # Initial case according to Zhou et al
    P_0_init = np.diag([
        10.0 ** 2, 10.0 ** 2, 10.0 ** 2,           # r  [km^2]
        1e-3 ** 2, 1e-3 ** 2, 1e-3 ** 2,              # v  [km^2/s^2]
        5e-3 ** 2, 5e-3 ** 2, 5e-3 ** 2,              # dv [km^2/s^2] (~5 m/s)
        50.0 ** 2,                                    # t1 [s^2]
    ])

    P_0_sparse = np.diag([10**2]*3 + [1e-3**2]*3 + [5e-3**2]*3 + [10**2])

    errors, converged_i, n_fails = [], [], 0
    diags = []

    for i in range(n_runs):

        if case == "std":
            prop = type_order(180, 10, P_0_init, rdm_seed=i, record=True)
        elif case == "sparse":
            prop = type_order(10, 180, P_0_sparse, rdm_seed=i, record=True)
        elif case == "short arc":
            prop = type_order(150, 2, P_0_init, t1_truth=151, t1_guess=161, t2=300, rdm_seed=i, record=True)
        elif case == "geo":
            prop = type_order(300, 60,                                        
                P_0_init=P_0_init,
                    observer_coe=(17993.0215, 0.7301, 0, 0, 0, 180),
                    target_coe  =(17993.0215, 0.7301, 0, 0, 0, 170),
                    t1_truth=4485.25, 
                    t1_guess=4535.25,
                    dv_truth=(0.0,   -1.4773091, 0.0),
                    dv_guess=(0.005, -1.4723091, 0.005))

        prop.do_calc(print_vals=False)

        # Check if converged
        if getattr(prop, "converged", True):
            errors.append(prop._error_row(prop.X_i))
            converged_i.append(getattr(prop, "converged_i", np.nan))
        else:
            n_fails += 1

        if (i + 1) % print_every == 0:
            print(f"{i + 1}/{n_runs} runs done ({n_fails} non-converged so far)\n")

    return errors, converged_i, n_fails, diags


if __name__ == "__main__":

    errors, converged_i, n_fails, diags = run_monte_carlo(PropagateSatelliteLinearMan, case = "sparse", n_runs=100)
    print("-------- LINEAR --------")
    print(f"Converged: {len(errors)} / {len(errors) + n_fails}\n")
    report_rmse(errors)   

    if len(converged_i):
        print(f"Iterations to converge: mean {np.nanmean(converged_i):.2f}, "
              f"median {np.nanmedian(converged_i):.0f}, max {np.nanmax(converged_i):.0f}")

    errors_s, converged_i_s, n_fails_s, diags = run_monte_carlo(PropagateSatelliteSecondMan, case = "sparse", n_runs=100)
    print("-------- SECOND ORDER --------")
    print(f"Converged: {len(errors_s)} / {len(errors_s) + n_fails_s}\n")
    report_rmse(errors_s)   

    if len(converged_i_s):
        print(f"Iterations to converge: mean {np.nanmean(converged_i_s):.2f}, "
                f"median {np.nanmedian(converged_i_s):.0f}, max {np.nanmax(converged_i_s):.0f}")
