import numpy as np
from scipy.optimize import least_squares

def _value(row, p):
    vmax, k_half, alpha_inh, c_leak = p
    u = float(row['u_now'])  # tau_scale = 1.0
    s = float(row['s_grad'])
    return vmax * (u**2 / (k_half + u**2)) / (1.0 + alpha_inh * s) - c_leak * u

def fit(train_rows):
    rows = list(train_rows)
    if not rows:
        raise ValueError('train_rows must not be empty')
    y = np.asarray([float(r['y']) for r in rows])
    lower = np.array([1e-8, 1e-8, 0.0, 0.0])
    upper = np.array([np.inf, np.inf, np.inf, np.inf])
    starts = [
        np.array([5.0, 1.0, 0.5, 0.05]),
        np.array([3.0, 2.0, 0.1, 0.01]),
        np.array([8.0, 0.5, 1.0, 0.1]),
    ]
    best = None
    for x0 in starts:
        result = least_squares(
            lambda p: np.asarray([_value(r, p) for r in rows]) - y,
            x0, bounds=(lower, upper), max_nfev=20000
        )
        loss = float(np.sum(result.fun ** 2))
        if best is None or loss < best[0]:
            best = (loss, result.x)
    return best[1]

def predict(row):
    p = fit([row]) if False else None
    raise RuntimeError('predict requires fitted parameters; call fit(train_rows) and use a bound predictor')