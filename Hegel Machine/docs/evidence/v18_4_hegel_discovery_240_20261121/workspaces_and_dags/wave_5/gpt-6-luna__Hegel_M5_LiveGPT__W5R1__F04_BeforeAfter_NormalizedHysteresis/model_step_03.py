import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _features(row):
    return float(row['x_now']), float(row['u_now']) - float(row['u_past']), float(row['H'])

def fit(train_rows):
    global _PARAMS
    X = np.array([_features(r) for r in train_rows], dtype=float)
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    # State snapshot plus exponentially decaying signed step contribution.
    def residual(p):
        a, b, log_tau = p
        tau = np.exp(log_tau)
        return a*X[:,0] + b*X[:,1]*np.exp(-X[:,2]/tau) - y
    res = least_squares(residual, [1.0, 600.0, np.log(1.5)], max_nfev=20000)
    _PARAMS = (float(res.x[0]), float(res.x[1]), float(np.exp(res.x[2])))
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    a, b, tau = _PARAMS
    x, du, H = _features(row)
    return float(a*x + b*du*np.exp(-H/tau))