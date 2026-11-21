import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _simulate(row, p):
    a_slow, b_couple, c_drive, tau_fast = p[:4]
    # The archive's memory-ODE mechanism uses a saturating, sink-modulated
    # siphon input; c_drive is its amplitude and p[4] is rho.
    rho = p[4]
    u = float(row['u_now'])
    s = float(row['s_grad'])
    H = max(0.0, float(row['H']))
    x = float(row['x_now'])
    z0 = row.get('mem_conv', 0.0)
    z = 0.0 if z0 is None else float(z0)
    J = c_drive * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    n = max(1, int(math.ceil(H / 0.004)))
    dt = H / n
    def deriv(state):
        xx, zz = state
        return np.array([-a_slow * xx + b_couple * zz + J,
                         (0.52 * xx - zz) / tau_fast], dtype=float)
    state = np.array([x, z], dtype=float)
    for _ in range(n):
        k1 = deriv(state)
        k2 = deriv(state + 0.5 * dt * k1)
        k3 = deriv(state + 0.5 * dt * k2)
        k4 = deriv(state + dt * k3)
        state += (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return float(state[0])

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([_simulate(r, p) for r in rows]) - y
    best = None
    # Multiple initial scales help with coupled-rate parameter correlations.
    for start in ([0.2, 0.1, 1.0, 1.0, 0.7], [0.05, 0.5, 2.0, 3.0, 0.3],
                  [0.5, 0.02, 3.0, 8.0, 0.9]):
        res = least_squares(residual, np.array(start, dtype=float),
            bounds=([1e-5, -5.0, -20.0, 0.02, -2.0],
                    [5.0, 5.0, 20.0, 100.0, 2.0]), max_nfev=1200)
        if best is None or np.sum(res.fun**2) < np.sum(best.fun**2):
            best = res
    _PARAMS = best.x
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return _simulate(row, _PARAMS)
