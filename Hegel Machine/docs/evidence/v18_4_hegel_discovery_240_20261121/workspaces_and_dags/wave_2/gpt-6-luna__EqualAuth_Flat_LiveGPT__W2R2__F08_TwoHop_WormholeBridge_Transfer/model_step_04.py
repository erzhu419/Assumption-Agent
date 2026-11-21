import numpy as np
from scipy.optimize import least_squares

_PARAMS = None
_U0 = 1.0
_S0 = 1.0

def _features(row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    du = u - _U0
    ds = s - _S0
    return np.array([1.0, du, du**2, du**3, ds, ds**2, du*ds, du**2*ds], dtype=float)

def fit(train_rows):
    global _PARAMS
    X = np.vstack([_features(r) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    # Numerical least-squares fit with mild regularization to discourage unsupported curvature.
    scale = max(float(np.std(y)), 1.0)
    def residual(p):
        return np.concatenate([X @ p - y, 1e-4 * p[2:]])
    result = least_squares(residual, np.zeros(X.shape[1]), max_nfev=5000)
    _PARAMS = result.x
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_features(row) @ _PARAMS)
