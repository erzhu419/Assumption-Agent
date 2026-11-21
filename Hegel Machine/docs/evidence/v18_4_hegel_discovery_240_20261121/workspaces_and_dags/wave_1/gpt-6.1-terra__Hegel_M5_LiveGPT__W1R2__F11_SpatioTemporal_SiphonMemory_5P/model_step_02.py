import numpy as np
from scipy.optimize import least_squares

_params = None

def _forecast(row, p):
    a, b, vmax, rho, tau = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    h = float(row['H'])
    drive = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    state = np.array([float(row['x_now']), float(row['mem_conv'])], dtype=float)
    steps = max(32, int(np.ceil(abs(h) / min(0.01, tau / 5.0))))
    dt = h / steps
    def deriv(q):
        x, z = q
        return np.array([-a*x + b*z + drive, (0.52*x - z)/tau])
    for _ in range(steps):
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
    bounds = ([0.0, 0.0, 1.0, 0.05, 0.02], [10.0, 10.0, 15.0, 0.95, 10.0])
    best = None
    for tau in (0.15, 0.6, 2.0):
        result = least_squares(residual, [1.0, 1.0, 6.5, 0.7, tau], bounds=bounds, max_nfev=1000, xtol=1e-11, ftol=1e-11, gtol=1e-11)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = best.x
    return _params

def predict(row):
    return _forecast(row, _params)
