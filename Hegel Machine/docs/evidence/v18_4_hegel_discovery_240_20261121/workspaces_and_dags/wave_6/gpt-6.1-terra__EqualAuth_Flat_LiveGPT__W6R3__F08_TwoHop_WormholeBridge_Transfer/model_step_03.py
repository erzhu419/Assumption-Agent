import numpy as np
from scipy.optimize import least_squares

_params = None

def _features(row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return np.array([u, u*u, u*s, u*u*u]), np.array([1.0, u, s, u*u, u*s, s*s])

def fit(train_rows):
    global _params
    rows = list(train_rows)
    nums, dens = zip(*[_features(r) for r in rows])
    N = np.asarray(nums)
    D = np.asarray(dens)
    y = np.asarray([float(r['y']) for r in rows])
    # Initialize the rational model by linearized least squares.
    A = np.column_stack((N, -y[:, None] * D[:, 1:]))
    p0 = np.linalg.lstsq(A, y, rcond=None)[0]
    def residual(p):
        den = 1.0 + D[:, 1:] @ p[4:]
        safe = np.where(np.abs(den) < 1e-8, np.where(den < 0, -1e-8, 1e-8), den)
        return np.r_[(N @ p[:4]) / safe - y, 1e-7 * p]
    result = least_squares(residual, p0, max_nfev=10000)
    _params = result.x

def predict(row):
    n, d = _features(row)
    den = 1.0 + d[1:] @ _params[4:]
    if abs(den) < 1e-8:
        den = -1e-8 if den < 0 else 1e-8
    return float((n @ _params[:4]) / den)
