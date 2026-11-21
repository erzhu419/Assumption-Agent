import numpy as np
from scipy.optimize import least_squares

_model = None

def fit(train_rows):
    global _model
    y = np.array([r['y'] for r in train_rows], dtype=float)
    u = np.array([r['u_ctrl'] for r in train_rows], dtype=float)
    best = None
    for key in ('x_ind', 'z_1', 'z_2', 'z_3', 'z_4'):
        x2 = np.array([r[key] ** 2 for r in train_rows], dtype=float)
        def residual(p):
            return p[0] * x2 / (p[1] + x2) - p[2] * u - y
        result = least_squares(residual, [8.0, 1.0, 0.2], bounds=([0.0, 1e-9, 0.0], [np.inf, np.inf, np.inf]), max_nfev=10000)
        score = float(np.dot(result.fun, result.fun))
        if best is None or score < best[0]:
            best = (score, key, result.x)
    _model = (best[1], best[2])

def predict(row):
    key, p = _model
    x2 = float(row[key]) ** 2
    return float(p[0] * x2 / (p[1] + x2) - p[2] * float(row['u_ctrl']))