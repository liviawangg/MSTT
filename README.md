# MSTT

Batch estimator for orbit determination and a **single impulsive manouevre** reconstruction of a target spacecraft, 
observed by **line-of-sight (LOS) unit vectors** from an observer satellite. This program implements the first and 
second order state transition tensors (STTs) with J2 dynamics, following the equations of Zhou et al.'s paper 
*Orbit Determination for Impulsively Maneuvering Spacecraft Using Modified State Transition Tensor*. 

The estimator solves for 10 parameters:

| Index | Parameter | Units |
|-------|-----------|-------|
| 0–2   | target position at `t0`, `r0` | km |
| 3–5   | target velocity at `t0`, `v0` | km/s |
| 6–8   | impulsive delta-v, `dv` | km/s |
| 9     | manoeuvre epoch, `t1` | s |

## Requirements

- Python 3.9+
- NumPy
- SciPy (`solve_ivp`, DOP853)

```bash
pip install numpy scipy
```

## Key Files
| File | Description |
|---------|---------|
| `second_order_2comp.py` | Proposed method, 2D tangent plane reformulation of 3D LOS measurement model, Second Order MSTT LSM | 
| `linear.py` | 3D LOS representation, Linear MSTT LSM |
| `calc_rmse_sum.py` | Given: estimator, orbit conditions and case; Provides a summary of statistics | 


## Key Function Parameters

Variables relating to orbit conditions have been described in the submitted paper.
| Setting | Default | Meaning |
|---------|---------|---------|
| `num_measurement`, `time_split` | — | Number of intervals and spacing (s); `t2 = t0 + num_measurement·time_split` |
| `P_0_init` | — | 10×10 initial covariance used in the weight matrix |
| `observer_coe`, `target_coe` | see code | `(alt km, e, i, RAAN, argp, ν)` in degrees. `a = (alt + R_e)/(1 + e)`, so `alt` is the apogee altitude |
| `t1_truth`, `dv_truth` | 905 s, 10 m/s per axis | True manoeuvre |
| `t1_guess`, `dv_guess` | 955 s, 15 m/s per axis | Initial guess (`t1_guess=None` → mid-arc) |
| `x0_offset` | 10 km, 1 m/s per axis | Initial-guess error on `[r0, v0]` |
| `project` | `True` | Project the LOS residual to 2 components |
| `j2_partials` | `True` | Include J2 in the STT partials (the dynamics always include J2) |
| `i_max` | 10 | Iteration limit |
| `rdm_seed` | 5 | Noise seed |
| `sigma_noise` (class) | 1e-5 | LOS noise, 1-σ |
| `nu` (class) | 1e-3 | Convergence tolerance |
| `tol_scale` (class) | `[10 km ×3, 1e-3 km/s ×3, 5e-3 km/s ×3, 50 s]` | Scaling for the normal equations and the convergence test |

### `project`
- `True`: use the proposed 2D tangent plane reformulation
- `False`: use Zhou et al.'s proposed 3D LOS measurement representation


## Use Example
```python
import numpy as np
from second_order_2comp import PropagateSatelliteSecondManProj

P0 = np.diag(PropagateSatelliteSecondManProj.tol_scale**2)   # 10x10 initial covariance

est = PropagateSatelliteSecondManProj(
    num_measurement=20,     # K = 21 measurement epochs
    time_split=90.0,        # s between measurements -> arc of 1800 s
    P_0_init=P0,
    t1_truth=905.0,
    t1_guess=955.0,
)
est.do_calc()               # prints the error vs truth each iteration

print(est.X_i)              # final estimate
print(np.sqrt(np.diag(est.P_2nd)))   # 1-sigma uncertainties
```
