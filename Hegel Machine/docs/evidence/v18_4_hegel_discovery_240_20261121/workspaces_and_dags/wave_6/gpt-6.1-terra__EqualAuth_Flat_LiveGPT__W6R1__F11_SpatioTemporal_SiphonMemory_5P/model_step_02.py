import numpy as np
from scipy.optimize import least_squares

_params = None

def _evaluate(rows, p):
    a, b, vmax, rho, tau = p
    x = np.array([r['x_now'] for r in rows], dtype=float)
    z = np.array([r['mem_conv'] for r in rows], dtype=float)
    h = np.array([r['H'] for r in rows], dtype=float) / 200.0
    u = np.array([r['u_now'] for r in rows], dtype=float) / 0.05
    s = np.array([r['s_grad'] for r in rows], dtype=float)
    j = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    def deriv(x, z):
        return -a * x + b * z + j, (0.52 * x - z) / tau
    for _ in range(200):
        k1x, k1z = deriv(x, z)
        k2x, k2z = deriv(x + h*k1x/2, z + h*k1z/2)
        k3x, k3z = deriv(x + h*k2x/2, z + h*k2z/2)
        k4x, k4z = deriv(x + h*k3x, z + h*k3z)
        x = x + h*(k1x + 2*k2x + 2*k3x + k4x)/6
        z = z + h*(k1z + 2*k2z + 2*k3z + k4z)/6
    return x

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.array([r['y'] for r in rows], dtype=float)
    lower = [0.0, 0.0, 0.0, 0.0, 0.005]
    upper = [10.0, 10.0, 20.0, 2.0, 2.0]
    def residual(p):
        return _evaluate(rows, p) - y
    results = [least_squares(residual, start, bounds=(lower, upper), max_nfev=1500, ftol=1e-11, xtol=1e-11, gtol=1e-11) for start in ([1.0, 1.0, 2.0, 0.5, 0.1], [2.0, 2.0, 4.0, 0.8, 0.3])]
    _params = min(results, key=lambda r: np.dot(r.fun, r.fun)).x
    return _params

def predict(row):
    return float(_evaluate([row], _params)[0])