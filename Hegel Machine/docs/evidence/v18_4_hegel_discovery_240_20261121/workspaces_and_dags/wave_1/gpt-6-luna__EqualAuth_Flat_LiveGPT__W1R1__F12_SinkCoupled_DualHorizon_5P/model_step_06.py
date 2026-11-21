import numpy as np
from scipy.optimize import least_squares

def _simulate(rows, p):
    a, b, c, tau = p
    H = np.array([float(r['H']) for r in rows])
    u = np.array([float(r['u_now']) for r in rows])
    x = np.array([float(r['x_now']) for r in rows])
    z = np.array([float(r.get('mem_conv') or 0.0) for r in rows])
    n = np.maximum(1, np.ceil(H / 0.08).astype(int))
    dt = H / n
    def deriv(xx, zz):
        return -a * xx + b * zz + c * u, (0.52 * xx - zz) / tau
    for k in range(int(n.max())):
        active = (k < n)
        d = dt * active
        k1x, k1z = deriv(x, z)
        k2x, k2z = deriv(x + 0.5*d*k1x, z + 0.5*d*k1z)
        k3x, k3z = deriv(x + 0.5*d*k2x, z + 0.5*d*k2z)
        k4x, k4z = deriv(x + d*k3x, z + d*k3z)
        x = x + d*(k1x + 2*k2x + 2*k3x + k4x)/6
        z = z + d*(k1z + 2*k2z + 2*k3z + k4z)/6
    return x

def fit(train_rows):
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return _simulate(train_rows, p) - y
    result = least_squares(residual, x0=np.array([0.1, 0.1, 1.0, 1.0]), bounds=([0.0001, -3.0, -5.0, 0.02], [3.0, 3.0, 5.0, 20.0]), max_nfev=1200)
    return result.x

_params = None
def predict(row):
    global _params
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_simulate([row], _params)[0])

_fit_original = fit
def fit(train_rows):
    global _params
    _params = _fit_original(train_rows)
    return _params