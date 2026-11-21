import numpy as np
from scipy.optimize import least_squares

_params = None

def _forecast(row, p):
    a, b, vmax, rho, tau = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    H = float(row['H'])
    drive = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    state = np.array([float(row['x_now']), float(row['mem_conv'])], dtype=float)
    n = max(1, int(np.ceil(abs(H) / 0.01)))
    dt = H / n
    def rhs(v):
        x, z = v
        return np.array([-a*x + b*z + drive, (0.52*x - z)/tau])
    for _ in range(n):
        k1 = rhs(state)
        k2 = rhs(state + 0.5*dt*k1)
        k3 = rhs(state + 0.5*dt*k2)
        k4 = rhs(state + dt*k3)
        state = state + dt*(k1 + 2*k2 + 2*k3 + k4)/6.0
    return float(state[0])

def fit(train_rows):
    global _params
    rows = list(train_rows)
    def residual(p):
        return np.array([_forecast(r, p) - float(r['y']) for r in rows])
    result = least_squares(residual, [0.8, 0.5, 6.5, 0.7, 0.5], bounds=([0.0, 0.0, 1.0, 0.05, 0.03], [10.0, 10.0, 15.0, 0.95, 10.0]), max_nfev=3000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
    _params = result.x
    return _params

def predict(row):
    return _forecast(row, _params)
