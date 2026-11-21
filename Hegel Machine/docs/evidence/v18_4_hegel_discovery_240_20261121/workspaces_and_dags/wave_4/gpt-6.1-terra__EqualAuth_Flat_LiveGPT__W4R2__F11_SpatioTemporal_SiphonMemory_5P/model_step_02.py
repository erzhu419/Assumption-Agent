import numpy as np
from scipy.optimize import least_squares

_params = None

def _forecast(row, p):
    a, b, vmax, rho, tau = p
    u = float(row['u_now']) / 2.0
    s = float(row['s_grad'])
    siphon = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    state = np.array([float(row['x_now']), float(row['mem_conv'])], dtype=float)
    H = float(row['H'])
    n = max(32, int(np.ceil(abs(H) / 0.01)))
    dt = H / n
    def deriv(v):
        x, z = v
        return np.array([-a*x + b*z + siphon, (0.52*x - z)/tau])
    for _ in range(n):
        k1 = deriv(state)
        k2 = deriv(state + 0.5*dt*k1)
        k3 = deriv(state + 0.5*dt*k2)
        k4 = deriv(state + dt*k3)
        state = state + dt*(k1 + 2*k2 + 2*k3 + k4)/6.0
    return float(state[0])

def fit(train_rows):
    global _params
    rows = list(train_rows)
    def residual(p):
        return np.array([_forecast(r, p) - float(r['y']) for r in rows])
    bounds = ([0.0, 0.0, 0.0, 0.0, 0.05], [5.0, 5.0, 10.0, 1.0, 10.0])
    starts = [[0.4, 0.7, 2.0, 0.5, 0.8], [1.0, 1.0, 3.0, 0.3, 2.0], [0.2, 0.3, 1.0, 0.8, 0.3]]
    results = [least_squares(residual, x, bounds=bounds, max_nfev=1200, ftol=1e-11, xtol=1e-11, gtol=1e-11) for x in starts]
    _params = min(results, key=lambda r: np.dot(r.fun, r.fun)).x
    return _params

def predict(row):
    return _forecast(row, _params)
