import math
import numpy as np
from scipy.optimize import least_squares

_params = None


def fit(train_rows):
    global _params
    h = np.array([r['H'] for r in train_rows], dtype=float)
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)

    def residual(log_params):
        af, a_s, tf, gap = np.exp(log_params)
        ts = tf + gap
        return u * (-af * np.expm1(-h / tf) + a_s * np.expm1(-h / ts)) - y

    starts = [(1.5, 2.5, 0.35, 8.0), (2.0, 3.0, 0.6, 3.0), (2.0, 4.0, 0.45, 20.0)]
    best = min((least_squares(residual, np.log(start), bounds=(-12.0, 12.0), x_scale='jac', ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=2000) for start in starts), key=lambda result: result.cost)
    _params = tuple(np.exp(best.x))


def predict(row):
    af, a_s, tf, gap = _params
    h = float(row['H'])
    return float(row['u_now'] * (-af * math.expm1(-h / tf) + a_s * math.expm1(-h / (tf + gap))))