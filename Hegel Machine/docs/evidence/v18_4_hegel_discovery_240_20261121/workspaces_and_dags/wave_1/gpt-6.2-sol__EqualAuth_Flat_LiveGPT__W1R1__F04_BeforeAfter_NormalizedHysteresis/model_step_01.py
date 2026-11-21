import math
import numpy as np
from scipy.optimize import least_squares

TAU_SCALE = 1.0
_params = None


def fit(train_rows):
    global _params
    data = np.asarray([(r['u_past'], r['u_now'], r['H'], r['y']) for r in train_rows], dtype=float)
    past, now, horizon, observed = data.T
    now = now / TAU_SCALE
    step = now - past / TAU_SCALE

    def residual(p):
        gain, rate, k_eq = p
        return gain * step * np.exp(-rate * horizon) + k_eq * now - observed

    result = least_squares(residual, [1.0, 1.0, 0.0], bounds=([-np.inf, 0.0, -np.inf], [np.inf, np.inf, np.inf]), ftol=1e-12, xtol=1e-12, gtol=1e-12)
    _params = result.x
    return _params


def predict(row) -> float:
    gain, rate, k_eq = _params
    now = row['u_now'] / TAU_SCALE
    step = now - row['u_past'] / TAU_SCALE
    return float(gain * step * math.exp(-rate * row['H']) + k_eq * now)
