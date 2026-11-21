import numpy as np
from scipy.optimize import least_squares

_params = None

def _features(rows):
    return np.array([[float(r['u_now']), float(r['s_grad'])] for r in rows], dtype=float)

def _model(p, X):
    u, s = X[:, 0], X[:, 1]
    lu = np.log1p(np.maximum(u, 0.0) * np.exp(p[4]))
    ls = np.log1p(np.maximum(s, 0.0) * np.exp(p[5]))
    return p[0] + p[1]*lu + p[2]*ls + p[3]*lu*ls

def fit(train_rows):
    global _params
    X = _features(train_rows)
    y = np.array([float(r['y']) for r in train_rows])
    best = None
    for ku in (0.3, 1.0, 3.0, 10.0):
        for ks in (0.3, 1.0, 3.0, 10.0):
            p0 = [0.0, 2.0, -1.0, 0.0, np.log(ku), np.log(ks)]
            result = least_squares(lambda p: _model(p, X) - y, p0, bounds=([-100, -100, -100, -100, -6, -6], [100, 100, 100, 100, 6, 6]), max_nfev=3000)
            if best is None or np.sum(result.fun**2) < np.sum(best.fun**2):
                best = result
    _params = best.x

def predict(row):
    return float(_model(_params, _features([row]))[0])