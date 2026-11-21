import numpy as np
from scipy.optimize import least_squares

_tau_scale = 2.0
_params = None

def _predict_with(row, p):
    a_slow, b_couple, tau_fast, vmax, rho = p
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    H = float(row['H'])
    u_n = float(row['u_now']) / _tau_scale
    s = float(row['s_grad'])
    J = vmax * (1.0 - rho * s / (1.0 + s)) * (u_n / (1.2 + u_n))
    def deriv(xv, zv):
        return -a_slow * xv + b_couple * zv + J, (0.52 * xv - zv) / tau_fast
    n = max(1, int(np.ceil(H / 0.004)))
    dt = H / n
    for _ in range(n):
        k1x, k1z = deriv(x, z)
        k2x, k2z = deriv(x + 0.5*dt*k1x, z + 0.5*dt*k1z)
        k3x, k3z = deriv(x + 0.5*dt*k2x, z + 0.5*dt*k2z)
        k4x, k4z = deriv(x + dt*k3x, z + dt*k3z)
        x += (dt/6.0) * (k1x + 2*k2x + 2*k3x + k4x)
        z += (dt/6.0) * (k1z + 2*k2z + 2*k3z + k4z)
    return x

def fit(train_rows):
    global _params
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_predict_with(r, p) for r in train_rows]) - y
    result = least_squares(residual, x0=[0.4, 0.2, 0.3, 1.0, 0.3], bounds=([0.0, 0.0, 0.005, 0.0, 0.0], [10.0, 10.0, 10.0, 20.0, 1.0]), max_nfev=3000)
    _params = result.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('Call fit(train_rows) before predict(row)')
    return float(_predict_with(row, _params))