import numpy as np
from scipy.optimize import least_squares

_params = None

def _evaluate(row, p):
    a, b, vmax, rho, tau = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    j = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    state = np.array([float(row['x_now']), float(row['mem_conv'])], dtype=float)
    h = float(row['H']) / 80.0
    def deriv(v):
        x, z = v
        return np.array([-a*x + b*z + j, (0.52*x - z)/tau])
    for _ in range(80):
        k1 = deriv(state)
        k2 = deriv(state + 0.5*h*k1)
        k3 = deriv(state + 0.5*h*k2)
        k4 = deriv(state + h*k3)
        state = state + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    return float(state[0])

def fit(train_rows):
    global _params
    def residual(p):
        return np.array([_evaluate(r, p) - float(r['y']) for r in train_rows])
    result = least_squares(residual, [0.5, 0.5, 6.5, 0.7, 1.0], bounds=([0.0, 0.0, 0.0, 0.0, 0.05], [10.0, 10.0, 30.0, 1.0, 20.0]), max_nfev=3000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
    _params = result.x
    return _params

def predict(row):
    return _evaluate(row, _params)
