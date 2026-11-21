import numpy as np
from scipy.optimize import least_squares

_fit_params = None

def _simulate(row, p):
    vmax, rho, a_slow, b_couple, tau_fast = p
    u_n = float(row['u_now'])  # tau_scale = 1.0
    s = float(row['s_grad'])
    H = float(row['H'])
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    j = vmax * (1.0 - rho * s / (1.0 + s)) * (u_n / (1.2 + u_n))
    n = max(20, int(np.ceil(H / 0.002)))
    dt = H / n
    def deriv(xv, zv):
        return -a_slow * xv + b_couple * zv + j, (0.52 * xv - zv) / tau_fast
    for _ in range(n):
        k1x, k1z = deriv(x, z)
        k2x, k2z = deriv(x + 0.5*dt*k1x, z + 0.5*dt*k1z)
        k3x, k3z = deriv(x + 0.5*dt*k2x, z + 0.5*dt*k2z)
        k4x, k4z = deriv(x + dt*k3x, z + dt*k3z)
        x += dt * (k1x + 2*k2x + 2*k3x + k4x) / 6.0
        z += dt * (k1z + 2*k2z + 2*k3z + k4z) / 6.0
    return x

def fit(train_rows):
    global _fit_params
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        return np.array([_simulate(r, p) for r in train_rows]) - y
    starts = [np.array([2.0, 0.3, 0.7, 0.5, 0.8]), np.array([3.0, 0.5, 1.0, 1.0, 1.2])]
    results = [least_squares(residual, p0, bounds=([0, 0, 0, 0, 0.05], [20, 1, 10, 10, 10]), max_nfev=3000) for p0 in starts]
    _fit_params = min(results, key=lambda res: np.sum(res.fun**2)).x
    return _fit_params

def predict(row) -> float:
    if _fit_params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_simulate(row, _fit_params))