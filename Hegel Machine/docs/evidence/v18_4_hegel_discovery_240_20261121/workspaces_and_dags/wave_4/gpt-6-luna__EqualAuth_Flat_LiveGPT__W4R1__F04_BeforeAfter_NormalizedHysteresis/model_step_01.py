import math
import numpy as np
from scipy.optimize import least_squares


def _value(row, p):
    gain, log_tau, k_eq = p
    tau = math.exp(log_tau)
    u_n = row['u_now']
    u_p = row['u_past']
    H = row['H']
    return gain * (u_n - u_p) * math.exp(-H / tau) + k_eq * u_n


def fit(train_rows):
    y = np.asarray([r['y'] for r in train_rows], dtype=float)
    def residuals(p):
        return np.asarray([_value(r, p) for r in train_rows]) - y
    result = least_squares(residuals, x0=np.array([2.0, 0.0, 0.5]))
    return result.x


def predict(row):
    return float(_value(row, _fit_parameters))