import numpy as np
from scipy.optimize import least_squares

_params = None
_anchor = None

def fit(train_rows):
    global _params, _anchor
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    raw = np.sqrt(np.array([float(r['u_raw']) for r in rows]))
    best = None
    for anchor in ('m_1', 'm_2', 'm_3', 's_surf'):
        x = np.array([float(r[anchor]) for r in rows])
        def residual(p):
            return p[0] * x / (p[1] + x) + p[2] * raw - y
        result = least_squares(residual, [8.0, 1.0, 0.3], bounds=([0.0, 1e-9, -np.inf], [np.inf, np.inf, np.inf]), max_nfev=5000)
        error = float(np.dot(result.fun, result.fun))
        if best is None or error < best[0]:
            best = (error, anchor, result.x)
    _, _anchor, _params = best

def predict(row):
    x = float(row[_anchor])
    return float(_params[0] * x / (_params[1] + x) + _params[2] * np.sqrt(float(row['u_raw'])))