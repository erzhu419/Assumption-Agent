import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None
_TAU_SCALE = 2.0

def _integrate(row, p):
    vmax, rho, a_slow, b_couple, tau_fast = p
    H = float(row['H'])
    x0 = float(row['x_now'])
    z0 = float(row['mem_conv'])
    u_n = float(row['u_now']) / _TAU_SCALE
    sg = float(row['s_grad'])
    siph = vmax * (1.0 - rho * sg / (1.0 + sg)) * (u_n / (1.2 + u_n))
    n = max(1, int(math.ceil(H / 0.005)))
    dt = H / n
    def deriv(state):
        x, z = state
        return np.array([-a_slow * x + b_couple * z + siph,
                         (0.52 * x - z) / tau_fast], dtype=float)
    state = np.array([x0, z0], dtype=float)
    for _ in range(n):
        k1 = deriv(state)
        k2 = deriv(state + 0.5 * dt * k1)
        k3 = deriv(state + 0.5 * dt * k2)
        k4 = deriv(state + dt * k3)
        state += (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
    return float(state[0])

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows], dtype=float)
    def residual(p):
        return np.asarray([_integrate(r, p) for r in rows]) - y
    starts = [np.array([3.0, 0.5, 0.5, 0.5, 1.0]),
              np.array([1.0, 0.2, 1.0, 1.0, 2.0]),
              np.array([5.0, 0.8, 0.2, 0.2, 0.5])]
    best = None
    for p0 in starts:
        result = least_squares(residual, p0, bounds=([0.0, 0.0, 0.0, 0.0, 0.02],
                                                       [20.0, 1.0, 10.0, 10.0, 20.0]),
                               max_nfev=3000)
        if best is None or np.sum(result.fun**2) < np.sum(best.fun**2):
            best = result
    _PARAMS = best.x.copy()
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    return _integrate(row, _PARAMS)