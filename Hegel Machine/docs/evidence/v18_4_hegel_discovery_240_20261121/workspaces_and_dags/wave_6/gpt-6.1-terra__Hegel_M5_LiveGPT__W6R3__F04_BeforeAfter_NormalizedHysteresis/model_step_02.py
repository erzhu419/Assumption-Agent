import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    rows = [r for r in train_rows if abs(r['u_now'] - r['u_past']) > 1e-12]
    h = np.array([r['H'] for r in rows], dtype=float)
    d = np.array([r['u_now'] - r['u_past'] for r in rows], dtype=float)
    z = np.array([(r['y'] - r['x_now']) / (r['u_now'] - r['u_past']) for r in rows], dtype=float)
    def residual(p):
        return (p[0] * np.exp(-p[1] * h) + p[2] * np.exp(-p[3] * h) - z) * np.abs(d)
    best = None
    for a, b in [(0.1, 1.0), (0.3, 2.0), (0.5, 4.0), (0.02, 0.8)]:
        result = least_squares(residual, [80.0, a, 100.0, b], bounds=([0, 0, 0, 0], [10000, 100, 10000, 100]), max_nfev=10000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = best.x

def predict(row):
    a, b, c, d = _params
    h = row['H']
    return float(row['x_now'] + (row['u_now'] - row['u_past']) * (a * np.exp(-b * h) + c * np.exp(-d * h)))