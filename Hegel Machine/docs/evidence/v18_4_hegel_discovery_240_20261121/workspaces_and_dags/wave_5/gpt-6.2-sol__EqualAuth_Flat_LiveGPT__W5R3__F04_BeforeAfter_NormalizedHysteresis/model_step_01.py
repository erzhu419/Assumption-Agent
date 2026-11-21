import math
import numpy as np
from scipy.optimize import least_squares

_params = None


def fit(train_rows):
    global _params
    rows = list(train_rows)
    u = np.array([r['u_now'] for r in rows], dtype=float) / 0.02
    d = u - np.array([r['u_past'] for r in rows], dtype=float) / 0.02
    h = np.array([r['H'] for r in rows], dtype=float)
    y = np.array([r['y'] for r in rows], dtype=float)
    tau0 = max(float(np.median(h)), 1e-3)
    design = np.column_stack((d * np.exp(-h / tau0), u))
    gain0, eq0 = np.linalg.lstsq(design, y, rcond=None)[0]

    def residual(p):
        gain, log_tau, eq = p
        return gain * d * np.exp(-h / math.exp(log_tau)) + eq * u - y

    _params = least_squares(residual, [gain0, math.log(tau0), eq0],
                            bounds=([-np.inf, -25, -np.inf], [np.inf, 25, np.inf])).x
    return _params


def predict(row) -> float:
    gain, log_tau, eq = _params
    u = row['u_now'] / 0.02
    d = (row['u_now'] - row['u_past']) / 0.02
    return float(gain * d * math.exp(-row['H'] / math.exp(log_tau)) + eq * u)
