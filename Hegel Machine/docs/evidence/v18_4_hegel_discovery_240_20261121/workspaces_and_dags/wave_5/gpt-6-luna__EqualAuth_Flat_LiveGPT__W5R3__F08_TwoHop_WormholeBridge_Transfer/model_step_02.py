import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    X = np.array([[float(r['u_now']), float(r['s_grad']), float(r['H']), float(r['u_past']), float(r['x_now']), float(r['mem_conv'])] for r in rows], dtype=float)
    y = np.array([float(r['y']) for r in rows], dtype=float)
    # Concave two-hop transfer response in current pull with a gradient-dependent
    # bridge correction. Fit all coefficients and the power numerically.
    def pred(p, x):
        u, s, h, up, xn, m = x.T
        return p[0] + p[1] * np.maximum(u, 1e-9) ** p[2] + p[3] * s + p[4] * s * u + p[5] * h + p[6] * up + p[7] * xn + p[8] * m
    def residual(p):
        return pred(p, X) - y
    p0 = np.array([0.0, 2.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    lo = [-100, -100, 0.05, -100, -100, -100, -100, -100, -100]
    hi = [100, 100, 1.5, 100, 100, 100, 100, 100, 100]
    result = least_squares(residual, p0, bounds=(lo, hi), max_nfev=20000)
    return result.x.tolist()

def predict(row):
    p = fit(_fit_rows)
    u = max(float(row['u_now']), 1e-9)
    s = float(row['s_grad']); h = float(row['H'])
    return float(p[0] + p[1]*u**p[2] + p[3]*s + p[4]*s*u + p[5]*h + p[6]*float(row['u_past']) + p[7]*float(row['x_now']) + p[8]*float(row['mem_conv']))

_fit_rows = []
# fit/predict API stores parameters from fit; callable predictors are handled by the harness.
_original_fit = fit
def fit(train_rows):
    global _fit_rows
    _fit_rows = list(train_rows)
    return _original_fit(_fit_rows)
