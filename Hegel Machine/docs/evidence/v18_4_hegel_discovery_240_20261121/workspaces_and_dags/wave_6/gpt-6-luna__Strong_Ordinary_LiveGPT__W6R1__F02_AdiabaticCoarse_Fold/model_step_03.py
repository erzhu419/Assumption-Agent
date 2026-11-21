import numpy as np
from scipy.optimize import least_squares

_params = None

def _simulate(rows, p):
    a, b, c, tau = p
    H = np.array([r['H'] for r in rows], dtype=float)
    x = np.array([r['x_now'] for r in rows], dtype=float)
    z = np.array([r['mem_conv'] for r in rows], dtype=float)
    u = np.array([r['u_now'] for r in rows], dtype=float)
    n = np.maximum(1, np.ceil(H / 0.01).astype(int))
    dt = H / n
    for j in range(int(n.max())):
        active = j < n
        h = dt * active
        def deriv(xx, zz):
            return -a * xx + b * zz + c * u, (0.52 * xx - zz) / tau
        k1x, k1z = deriv(x, z)
        k2x, k2z = deriv(x + 0.5*h*k1x, z + 0.5*h*k1z)
        k3x, k3z = deriv(x + 0.5*h*k2x, z + 0.5*h*k2z)
        k4x, k4z = deriv(x + h*k3x, z + h*k3z)
        x = x + (h/6.0)*(k1x + 2*k2x + 2*k3x + k4x)
        z = z + (h/6.0)*(k1z + 2*k2z + 2*k3z + k4z)
    return x

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.array([r['y'] for r in rows], dtype=float)
    def residual(p):
        return _simulate(rows, p) - y
    result = least_squares(residual, x0=[0.25, 0.08, 1.0, 0.25], bounds=([0.001, 0.0, 0.0, 0.01], [3.0, 3.0, 5.0, 3.0]), max_nfev=120)
    _params = result.x
    return _params

def predict(row):
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return float(_simulate([row], _params)[0])