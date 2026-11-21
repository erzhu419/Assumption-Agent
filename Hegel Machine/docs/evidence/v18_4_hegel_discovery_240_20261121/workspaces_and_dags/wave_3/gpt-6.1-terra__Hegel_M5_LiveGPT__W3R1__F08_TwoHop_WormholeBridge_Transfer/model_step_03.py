import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(p, u, s):
    a, b, c, d, e, f = p
    return a*u/(1.0 + b*u + c*s) + d*u/(1.0 + e*u + f*s)

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    rng = np.random.default_rng(42)
    starts = [np.array([4., 1., 0.2, 4., 0.5, 1.])]
    starts += [np.exp(rng.uniform(-2.0, 2.0, 6)) for _ in range(20)]
    best = None
    for start in starts:
        result = least_squares(lambda p: _model(p, u, s) - y, start,
                               bounds=(0.0, 1000.0), max_nfev=5000,
                               ftol=1e-11, xtol=1e-11, gtol=1e-11)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = best.x

def predict(row):
    return float(_model(_params, float(row['u_now']), float(row['s_grad'])))