import numpy as np
from scipy.optimize import least_squares

_params = None
_features = None

def _base(row):
    return (float(row['u_now']) - float(row['u_past'])) / 0.004

def _terms(row):
    h = float(row['H'])
    x = float(row['x_now'])
    d = _base(row)
    return np.array([1.0, h, h*h, x - 1.0, d, h*(x - 1.0), h*d])

def _fit_model(rows, idx):
    X = np.array([_terms(r)[idx] for r in rows])
    d = np.array([_base(r) for r in rows])
    x = np.array([float(r['x_now']) for r in rows])
    y = np.array([float(r['y']) for r in rows])
    p0 = np.zeros(len(idx)); p0[0] = np.log(3.0)
    if 1 in idx:
        p0[list(idx).index(1)] = -0.8
    def residual(p):
        z = np.clip(X @ p, -20, 20)
        return x + d * np.exp(z) - y
    return least_squares(residual, p0, max_nfev=3000).x

def fit(train_rows):
    global _params, _features
    rows = list(train_rows)
    candidates = [[0, 1], [0, 1, 2], [0, 1, 3], [0, 1, 4], [0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 3, 4], [0, 1, 3, 5], [0, 1, 4, 6]]
    best = None
    for idx in candidates:
        if len(rows) <= len(idx) + 1:
            continue
        errors = []
        try:
            for i, r in enumerate(rows):
                if abs(_base(r)) < 1e-12:
                    continue
                p = _fit_model(rows[:i] + rows[i+1:], idx)
                z = np.clip(_terms(r)[idx] @ p, -20, 20)
                pred = float(r['x_now']) + _base(r) * np.exp(z)
                errors.append((pred - float(r['y']))**2)
            score = np.mean(errors)
            if np.isfinite(score) and (best is None or score < best[0]):
                best = (score, idx)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            continue
    _features = best[1] if best is not None else [0, 1]
    _params = _fit_model(rows, _features)

def predict(row):
    z = np.clip(_terms(row)[_features] @ _params, -20, 20)
    return float(float(row['x_now']) + _base(row) * np.exp(z))
