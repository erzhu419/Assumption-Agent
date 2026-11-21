import numpy as np
from scipy.optimize import least_squares

_params = None

def _forecast(row, p):
    a, b, vmax, rho, tau = p
    u = float(row['u_now']) / 2.0
    s = float(row['s_grad'])
    drive = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    state = np.array([float(row['x_now']), float(row['mem_conv'])], dtype=float)
    H = float(row['H'])
    n = max(32, int(np.ceil(abs(H) / 0.01)))
    dt = H / n
    def deriv(v):
        x, z = v
        return np.array([-a*x + b*z + drive, (0.52*x - z)/tau])
    for _ in range(n):
        k1 = deriv(state)
        k2 = deriv(state + 0.5*dt*k1)
        k3 = deriv(state + 0.5*dt*k2)
        k4 = deriv(state + dt*k3)
        state = state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    return float(state[0])

def fit(train_rows):
    global _params
    rows = list(train_rows)
    def residual(p):
        return np.array([_forecast(r, p) - float(r['y']) for r in rows])
    bounds = ([0.0, 0.0, 0.0, 0.0, 0.05], [10.0, 10.0, 20.0, 1.0, 10.0])
    starts = [[0.5, 0.5, 6.5, 0.7, t] for t in (0.15, 0.5, 2.0, 6.0)]
    results = [least_squares(residual, p, bounds=bounds, max_nfev=1200, ftol=1e-12, xtol=1e-12, gtol=1e-12) for p in starts]
    _params = min(results, key=lambda result: np.dot(result.fun, result.fun)).x

def predict(row):
    return _forecast(row, _params)
