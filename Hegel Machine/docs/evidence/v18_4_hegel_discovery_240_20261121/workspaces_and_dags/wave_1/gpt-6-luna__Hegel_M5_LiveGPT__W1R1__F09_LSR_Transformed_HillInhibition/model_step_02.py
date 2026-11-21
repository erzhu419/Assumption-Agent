import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(row, p):
    vmax, k_half, alpha_inh, c_leak = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return vmax * (u*u / (k_half + u*u)) / (1.0 + alpha_inh*s) - c_leak*u

def fit(train_rows):
    global _params
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    # Positive parameters; use the supplied subtree warm-start ranges where applicable.
    best = None
    for start in ([6.5, 1.0, 0.7, 0.1], [5.0, 2.0, 0.4, 0.3], [8.0, 0.5, 1.0, 0.0]):
        result = least_squares(
            lambda p: np.asarray([_model(r, p) for r in train_rows]) - y,
            x0=np.asarray(start, dtype=float),
            bounds=([0.01, 1e-5, 0.0, 0.0], [30.0, 100.0, 20.0, 10.0]),
            max_nfev=20000
        )
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = best.x
    return {'vmax': float(_params[0]), 'k_half': float(_params[1]), 'alpha_inh': float(_params[2]), 'c_leak': float(_params[3])}

def predict(row):
    if _params is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    return float(_model(row, _params))