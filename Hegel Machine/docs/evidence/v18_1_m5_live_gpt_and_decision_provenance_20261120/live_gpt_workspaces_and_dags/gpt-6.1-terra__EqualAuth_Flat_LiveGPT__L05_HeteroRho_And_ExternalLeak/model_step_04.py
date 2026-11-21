import numpy as np
from scipy.optimize import least_squares

_params = None
_power = None

def _values(rows):
    u = np.array([r['u_now'] for r in rows], dtype=float)
    s = np.array([r['s_grad'] for r in rows], dtype=float)
    return u, s

def _model(p, u, s, power):
    vmax, rho, leak, threshold = p
    basin = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    return basin - leak * np.maximum(u - threshold, 0.0) ** power

def fit(train_rows):
    global _params, _power
    u, s = _values(train_rows)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    candidates = []
    for power in (1, 2):
        for threshold in (1.5, 2.0, 2.4, 2.8, 3.2):
            result = least_squares(
                lambda p: _model(p, u, s, power) - y,
                [4.0, 0.7, 0.5, threshold],
                bounds=([0.0, 0.0, 0.0, 0.0], [20.0, 2.0, 20.0, 4.0]),
                max_nfev=5000
            )
            candidates.append((float(np.sum(result.fun ** 2)), power, result.x))
    _, _power, _params = min(candidates, key=lambda item: item[0])

def predict(row):
    u, s = _values([row])
    return float(_model(_params, u, s, _power)[0])
