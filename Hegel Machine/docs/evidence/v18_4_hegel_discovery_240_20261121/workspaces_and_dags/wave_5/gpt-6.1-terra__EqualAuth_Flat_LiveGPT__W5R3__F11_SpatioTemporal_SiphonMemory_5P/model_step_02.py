import numpy as np
from scipy.optimize import least_squares

_params = None

def _forecast(row, p):
    a, b, tau, vmax, rho = p
    u = float(row['u_now']) / 2.0
    s = float(row['s_grad'])
    j = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    h = float(row['H'])
    n = max(32, int(abs(h) * 100))
    dt = h / n
    v = np.array([float(row['x_now']), float(row['mem_conv'])], dtype=float)
    def f(q):
        x, z = q
        return np.array([-a*x + b*z + j, (0.52*x - z)/tau])
    for _ in range(n):
        k1 = f(v)
        k2 = f(v + 0.5*dt*k1)
        k3 = f(v + 0.5*dt*k2)
        k4 = f(v + dt*k3)
        v = v + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    return float(v[0])

def fit(train_rows):
    global _params
    rows = list(train_rows)
    def residual(p):
        return np.array([_forecast(r, p) - float(r['y']) for r in rows])
    bounds = ([0.0, 0.0, 0.03, 0.0, 0.0], [10.0, 10.0, 20.0, 20.0, 1.0])
    starts = [[0.4, 0.7, 0.6, 2.0, 0.5], [0.2, 1.0, 2.0, 3.0, 0.8], [1.0, 0.5, 0.2, 3.0, 0.3]]
    fits = [least_squares(residual, p, bounds=bounds, max_nfev=1500, xtol=1e-11, ftol=1e-11, gtol=1e-11) for p in starts]
    _params = min(fits, key=lambda r: np.dot(r.fun, r.fun)).x

def predict(row):
    return _forecast(row, _params)
