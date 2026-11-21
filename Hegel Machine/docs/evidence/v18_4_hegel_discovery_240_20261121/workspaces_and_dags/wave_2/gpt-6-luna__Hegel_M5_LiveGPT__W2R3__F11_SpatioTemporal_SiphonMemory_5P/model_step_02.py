import math
import numpy as np
from scipy.optimize import least_squares

_TAU_SCALE = 0.05
_PARAMS = None

def _simulate(row, p):
    a_slow, b_couple, vmax, rho, tau_fast = p
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    u_n = float(row['u_now']) / _TAU_SCALE
    s = float(row['s_grad'])
    H = float(row['H'])
    drive = vmax * (1.0 - rho * s / (1.0 + s)) * (u_n / (1.2 + u_n))
    n = max(40, int(math.ceil(H / 0.002)))
    dt = H / n
    def deriv(state):
        xx, zz = state
        return np.array([-a_slow * xx + b_couple * zz + drive,
                         (0.52 * xx - zz) / tau_fast], dtype=float)
    state = np.array([x, z], dtype=float)
    for _ in range(n):
        k1 = deriv(state)
        k2 = deriv(state + 0.5 * dt * k1)
        k3 = deriv(state + 0.5 * dt * k2)
        k4 = deriv(state + dt * k3)
        state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return float(state[0])

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    y = np.array([r['y'] for r in rows], dtype=float)
    lo = np.array([0.01, 0.0, 0.1, 0.0, 0.003])
    hi = np.array([8.0, 8.0, 20.0, 0.999, 0.5])
    starts = [np.array([1.0, 1.0, 6.0, 0.70, 0.05]),
              np.array([0.5, 0.4, 5.0, 0.5, 0.02]),
              np.array([2.0, 2.0, 8.0, 0.85, 0.1])]
    best = None
    for x0 in starts:
        res = least_squares(lambda p: np.array([_simulate(r, p) for r in rows]) - y,
                            np.clip(x0, lo + 1e-8, hi - 1e-8), bounds=(lo, hi),
                            max_nfev=3000, x_scale='jac')
        score = float(np.dot(res.fun, res.fun))
        if best is None or score < best[0]:
            best = (score, res.x)
    _PARAMS = best[1]
    return _PARAMS.copy()

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return _simulate(row, _PARAMS)
