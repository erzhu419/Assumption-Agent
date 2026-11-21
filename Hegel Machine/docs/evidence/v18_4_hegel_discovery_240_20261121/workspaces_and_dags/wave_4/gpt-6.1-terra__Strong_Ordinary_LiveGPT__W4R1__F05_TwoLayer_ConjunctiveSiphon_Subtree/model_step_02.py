import numpy as np
from scipy.optimize import least_squares

_model = None
_params = None

def _features(row):
    return float(row['u_now']) / 4.0, float(row['s_grad'])

def _eval(name, p, row):
    u, s = _features(row)
    if name == 'sqrt':
        return p[0] * u / (1.2 + u) - p[1] * s / (1.0 + s) * u / (1.2 + u) + p[2] * np.sqrt(max(u, 0.0))
    if name == 'linear':
        return p[0] * u / (1.2 + u) - p[1] * s / (1.0 + s) * u / (1.2 + u) + p[2] * u
    if name == 'hill':
        return p[0] * u * u / (p[1] + u * u) / (1.0 + p[2] * s) - p[3] * u
    return p[0] * u * np.exp(-p[1] / (0.5 + u)) * (1.0 - p[2] * s / (1.0 + s)) + p[3] * np.sqrt(max(u, 0.0))

def fit(train_rows):
    global _model, _params
    y = np.array([r['y'] for r in train_rows], dtype=float)
    candidates = []
    for name in ('sqrt', 'linear'):
        X = np.array([[_eval(name, np.eye(3)[i], r) for i in range(3)] for r in train_rows])
        p = np.linalg.lstsq(X, y, rcond=None)[0]
        rss = np.sum((X @ p - y) ** 2)
        candidates.append((rss, name, p))
    # Prefer the simpler archived siphon branches when their residual is at rounding level.
    best = min(candidates, key=lambda item: item[0])
    if best[0] > 1e-10:
        for name, initial, bounds in [
            ('hill', [5.0, 1.0, 0.5, 0.1], ([0, 1e-8, 0, -np.inf], [np.inf, np.inf, np.inf, np.inf])),
            ('barrier', [2.0, 1.0, 0.7, 1.0], ([0, 0, 0, -np.inf], [np.inf, np.inf, 1.0, np.inf]))
        ]:
            result = least_squares(lambda p: np.array([_eval(name, p, r) for r in train_rows]) - y, initial, bounds=bounds, max_nfev=10000)
            rss = float(np.sum(result.fun ** 2))
            candidates.append((rss, name, result.x))
        best = min(candidates, key=lambda item: item[0])
    _, _model, _params = best

def predict(row):
    return float(_eval(_model, _params, row))