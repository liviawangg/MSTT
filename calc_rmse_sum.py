from linear_corrected import PropagateSatelliteLinearMan
from second_order_2comp import PropagateSatelliteSecondManProj

from functools import partial
import numpy as np
import time

# Second-order solver from second_order_2comp: project=False uses the full 3-component
# LOS vector, project=True projects it onto the 2D plane normal to the LOS
PropagateSatelliteSecondMan = partial(PropagateSatelliteSecondManProj, project=False)
PropagateSatelliteSecondManProjOn = partial(PropagateSatelliteSecondManProj, project=True)

COLS = ["x", "y", "z", "vx", "vy", "vz", "dvx", "dvy", "dvz", "t1"]

MD_MEDIAN_IDEAL = 3.08
MD_95_BOUND = 4.28
PAPER_MODE = True

def _fmt(v, width=10, prec=4):
    """Fixed-point, unless a non-zero value would round to 0.0000 -> show it in sci notation."""
    s = f"{v:{width}.{prec}f}"
    if v != 0 and float(s) == 0:
        s = f"{v:{width}.{max(prec - 2, 1)}e}"
    return s

def _row(label, values, width=10, prec=4):
    print(f"{label:<6}" + "".join(_fmt(v, width, prec) for v in values))

# Get the mean, std and rmse of each component from converged runs
def report_stats(errors):
    E = np.asarray(errors)                          
    mean = E.mean(axis=0)
    std = E.std(axis=0, ddof=1)                     
    rmse = np.sqrt(np.mean(E**2, axis=0))

    print(f"Statistics over {len(E)} converged runs")
    print(f"{'':6}{'Position, km':^30}{'Velocity, m/s':^30}{'Impulsive vector, m/s':^30}{'Epoch, s':^10}")
    print(f"{'':6}" + "".join(f"{c:>10}" for c in COLS))
    _row("Mean", mean)
    _row("STD", std)
    _row("RMSE", rmse)
    return mean, std, rmse

# Get the manoeuvre errors 
def report_maneuver(errors, dv_true_kms):
    E = np.asarray(errors)
    dv_true = np.asarray(dv_true_kms) * 1e3         
    dv_est = E[:, 6:9] + dv_true                   

    def alpha(v):
        return np.degrees(np.arcsin(v[..., 2] / np.linalg.norm(v, axis=-1)))

    def beta(v):
        return np.degrees(np.arctan2(v[..., 1], v[..., 0]))

    def wrap(a):                                    
        return (a + 180.0) % 360.0 - 180.0

    metrics = {
        "t1 [s]":      E[:, 9],
        # 
        "|dv| [m/s]":  np.linalg.norm(dv_est, axis=1) - np.linalg.norm(dv_true),
        # 
        "alpha [deg]": wrap(alpha(dv_est) - alpha(dv_true)),
        #
        "beta [deg]":  wrap(beta(dv_est) - beta(dv_true)),
    }

    print("Maneuver reconstruction errors")
    print(f"{'':12}{'Mean':>12}{'STD':>12}")
    for name, e in metrics.items():
        print(f"{name:12}{_fmt(e.mean(), 12)}{_fmt(e.std(ddof=1), 12)}")
    return metrics


# Get the mahalanobis distance
def mahalanobis(e, P):
    if P is None:
        return np.nan
    try:
        # sqrt(e^T P^-1 e)
        return float(np.sqrt(e @ np.linalg.solve(P, e)))
    except np.linalg.LinAlgError:
        return np.nan

# Get the mahalanobis distance
def report_md(md, label):
    md = np.asarray(md, float)
    md = md[np.isfinite(md)]
    if md.size == 0:
        print(f"{label}: no Mahalanobis distances available")
        return
    print(f"{label}: median MD {np.median(md):.2f} (ideal {MD_MEDIAN_IDEAL}), "
          f"within 95% bound {MD_95_BOUND}: {(md < MD_95_BOUND).mean():.1%} (ideal 95%), "
          f"mean MD^2 {np.mean(md**2):.2f} (ideal 10)")


def make_prop(type_order, case, seed):
    P_0_init = np.diag([10.0**2] * 3 + [1e-3**2] * 3 + [5e-3**2] * 3 + [50.0**2])
    P_0_sparse = np.diag([10.0**2] * 3 + [1e-3**2] * 3 + [5e-3**2] * 3 + [10.0**2])

    if case == "std":
        return type_order(180, 10, P_0_init, rdm_seed=seed, paper_mode=PAPER_MODE, record=True)
    if case == "sparse":
        return type_order(10, 180, P_0_sparse, t1_guess=915, rdm_seed=seed, paper_mode=PAPER_MODE, record=True)
    if case == "short arc":
        return type_order(150, 2, P_0_init, t1_truth=151, t1_guess=161, t2=300,
                          rdm_seed=seed, paper_mode=PAPER_MODE, record=True)
    if case == "geo":
        return type_order(300, 60, P_0_init=P_0_init,
                          observer_coe=(17993.0215, 0.7301, 0, 0, 0, 180),
                          target_coe=(17993.0215, 0.7301, 0, 0, 0, 170),
                          t1_truth=4485.25, t1_guess=4535.25,
                          dv_truth=(0.0, -1.4773091, 0.0),
                          dv_guess=(0.005, -1.4723091, 0.005),
                          rdm_seed=seed, paper_mode=PAPER_MODE, record=True)
    raise ValueError(f"Unknown case: {case}")


def run_monte_carlo(type_order, case="std", n_runs=100, print_every=10):
    errors, converged_i, n_fails, n_crash = [], [], 0, 0
    md_lin, md_2nd = [], []
    dv_true = None
    times, times_per_iter = [], []

    for i in range(n_runs):
        prop = make_prop(type_order, case, seed=i)
        dv_true = prop.dv_truth

        t_start = time.perf_counter()
        try:
            prop.do_calc(print_vals=False)
        except (RuntimeError, np.linalg.LinAlgError):
            # divergence, failed integration or singular matrix: count as non-converged
            n_fails += 1
            n_crash += 1
            continue
        elapsed = time.perf_counter() - t_start

        if getattr(prop, "converged", False):
            errors.append(prop._error_row(prop.X_i))
            converged_i.append(prop.converged_i)

            times.append(elapsed)
            times_per_iter.append(elapsed / prop.converged_i)

            # Mahalanobis distance, in the estimator's internal units (km, km/s, s)
            X_true = np.concatenate([prop.target[0], prop.dv_truth, [prop.t1_truth]])
            e = prop.X_i - X_true
            md_lin.append(mahalanobis(e, getattr(prop, "P_lin", None)))
            md_2nd.append(mahalanobis(e, getattr(prop, "P_2nd", None)))
        else:
            n_fails += 1

        # if (i + 1) % print_every == 0:
        #     print(f"{i + 1}/{n_runs} runs done ({n_fails} non-converged, {n_crash} crashed)")

    return errors, converged_i, n_fails, dv_true, md_lin, md_2nd, times, times_per_iter


def summarise(label, case, errors, converged_i, n_fails, dv_true, md_lin, md_2nd, times, times_per_iter):
    print(f"\n-------- {label} --------")
    print(f"Case: {case}")
    print(f"Converged: {len(errors)} / {len(errors) + n_fails}")
    if not errors:
        return
    report_stats(errors)
    report_maneuver(errors, dv_true)
    print(f"Iterations to converge: mean {np.mean(converged_i):.2f}, "
          f"median {np.median(converged_i):.0f}, max {np.max(converged_i):.0f}")
    report_md(md_lin, "Linear P")
    report_md(md_2nd, "Second-order P")

    t = np.asarray(times)
    print(f"Runtime: mean {t.mean():.4f} s, min {t.min():.4f} s, max {t.max():.4f} s, "
      f"per iteration {np.mean(times_per_iter):.3f} s")


# Box plot mahalanobis distance 
def plot_md(md_lin, md_2nd, title=""):
    import matplotlib.pyplot as plt
    data = [np.asarray(m, float)[np.isfinite(m)] for m in (md_2nd, md_lin)]
    plt.boxplot(data, labels=["Second-order P", "Linear P"])
    plt.axhline(np.sqrt(10), ls="--", c="g", label=r"$\sqrt{n_X}$ (1$\sigma$)")
    plt.axhline(3 * np.sqrt(10), ls="--", c="m", label=r"$3\sqrt{n_X}$ (3$\sigma$)")
    plt.ylabel("Mahalanobis distance")
    plt.title(title)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    case = "geo"
    n_runs = 100

    # res = run_monte_carlo(PropagateSatelliteLinearMan, case, n_runs)
    # summarise("LINEAR", case, *res)

    # res_s = run_monte_carlo(PropagateSatelliteSecondMan, case, n_runs)
    # summarise("SECOND ORDER", case, *res_s)

    res_s_proj = run_monte_carlo(PropagateSatelliteSecondManProjOn, case, n_runs)
    summarise("SECOND ORDER PROJECTION", case, *res_s_proj)

    # plot_md(res_s[4], res_s[5], title=f"Second-order solver, {case} case")