import numpy as np
from scipy.optimize import least_squares

_params = None

def _forecast(row, p):
    a, b, vmax, rho, tau = p
    u = float(row['u_now']) / 2.0
    s = float(row['s_grad'])
    drive = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    state = np.array([float(row['x_now']), float(row['mem_conv'])], dtype=float)
    h = float(row['H']) / 32.0
    def rhs(v):
        x, z = v
        return np.array([-a*x + b*z + drive, (0.52*x - z)/tau])
    for _ in range(32):
        k1 = rhs(state)
        k2 = rhs(state + 0.5*h*k1)
        k3 = rhs(state + 0.5*h*k2)
        k4 = rhs(state + h*k3)
        state = state + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    return float(state[0])

def fit(train_rows):
    global _params
    rows = list(train_rows)
    def residual(p):
        return np.array([_forecast(r, p) - float(r['y']) for r in rows])
    lower = [0.0, 0.0, 0.0, 0.0, 0.03]
    upper = [10.0, 10.0, 20.0, 1.0, 20.0]
    starts = [[0.5, 0.5, 6.5, 0.7, 0.5], [1.0, 1.0, 6.5, 0.7, 2.0], [0.2, 0.2, 6.5, 0.7, 0.15]]
    results = [least_squares(residual, x, bounds=(lower, upper), max_nfev=1500, xtol=1e-11, ftol=1e-11, gtol=1e-11) for x in starts]
    _params = min(results, key=lambda r: np.dot(r.fun, r.fun)).x
    return _params

def predict(row):
    if _params is None:
        raise RuntimeError('fit must be called before predict')
    return _forecast(row, _params)
