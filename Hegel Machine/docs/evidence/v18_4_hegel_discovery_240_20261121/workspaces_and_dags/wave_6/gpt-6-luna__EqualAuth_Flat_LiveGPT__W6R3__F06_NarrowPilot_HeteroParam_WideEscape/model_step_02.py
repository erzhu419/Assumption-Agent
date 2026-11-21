import math
import numpy as np
from scipy.optimize import least_squares


def _features(row):
    u = max(1e-6, float(row['u_now']))
    s = float(row['s_grad'])
    return u, s


def fit(train_rows):
    rows = list(train_rows)
    us = np.array([_features(r)[0] for r in rows], dtype=float)
    ss = np.array([_features(r)[1] for r in rows], dtype=float)
    ys = np.array([float(r['y']) for r in rows], dtype=float)

    # Saturating response with a linear gradient-dependent amplitude and
    # a local square-root contribution. Fit all parameters to the observations.
    def residual(p):
        amp, grad_slope, km, local = p
        pred = (amp - grad_slope * ss) * us / (km + us) + local * np.sqrt(us)
        return pred - ys

    result = least_squares(residual, x0=np.array([4.0, 1.0, 1.2, 0.2]),
                           bounds=([-100.0, -100.0, 1e-6, -100.0],
                                   [100.0, 100.0, 100.0, 100.0]),
                           max_nfev=20000)
    return result.x


def predict(row):
    p = fit(_CURRENT_TRAIN_ROWS)
    u, s = _features(row)
    amp, grad_slope, km, local = p
    return float((amp - grad_slope * s) * u / (km + u) + local * math.sqrt(u))