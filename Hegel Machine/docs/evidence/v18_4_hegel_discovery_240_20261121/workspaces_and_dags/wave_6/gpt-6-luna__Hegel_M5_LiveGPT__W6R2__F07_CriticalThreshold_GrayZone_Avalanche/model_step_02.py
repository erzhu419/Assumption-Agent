import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    vmax, rho, c_loc, k_aval, s_crit = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    u_n = u  # tau_scale = 1.0
    return (vmax * (1.0 - rho * s / (1.0 + s)) * (u_n / (1.2 + u_n))
            + c_loc * np.sqrt(max(u_n, 0.0))
            - k_aval * max(0.0, s - s_crit) * u_n)

def fit(train_rows):
    global _params
    y = np.asarray([float(r['y']) for r in train_rows])
    starts = [
        [6.5, 0.70, 0.1, 1.0, 2.58],
        [7.0, 0.68, 0.5, 1.5, 2.65],
        [5.5, 0.74, -0.3, 2.0, 2.50],
        [8.0, 0.72, 0.0, 0.8, 2.60],
    ]
    lower = [1.0, 0.05, -2.0, 0.0, 2.3]
    upper = [15.0, 0.95, 2.0, 10.0, 2.9]
    best = None
    for start in starts:
        result = least_squares(
            lambda p: np.asarray([_value(r, p) for r in train_rows]) - y,
            x0=np.asarray(start, dtype=float), bounds=(lower, upper),
            max_nfev=20000
        )
        score = float(np.dot(result.fun, result.fun))
        if best is None or score < best[0]:
            best = (score, result.x)
    _params = best[1]
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_value(row, _params))