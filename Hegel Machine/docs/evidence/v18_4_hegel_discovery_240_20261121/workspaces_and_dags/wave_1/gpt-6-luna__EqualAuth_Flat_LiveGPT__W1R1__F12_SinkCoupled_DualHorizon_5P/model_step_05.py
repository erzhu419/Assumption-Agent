import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _simulate(row, p):
    a_slow, b_couple, c_drive, tau_fast = p
    H = max(0.0, float(row['H']))
    n = max(1, int(math.ceil(H / 0.004)))
    dt = H / n if n else 0.0
    x = float(row['x_now'])
    z0 = row.get('mem_conv')
    z = float(z0) if z0 is not None else 0.52 * x
    u = float(row['u_now'])
    def f(state):
        xx, zz = state
        return np.array([-a_slow * xx + b_couple * zz + c_drive * u,
                         (0.52 * xx - zz) / tau_fast])
    state = np.array([x, z], dtype=float)
    for _ in range(n):
        k1 = f(state)
        k2 = f(state + 0.5 * dt * k1)
        k3 = f(state + 0.5 * dt * k2)
        k4 = f(state + dt * k3)
        state += (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return float(state[0])

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    def residual(q):
        p = [math.exp(q[0]), q[1], q[2], math.exp(q[3])]
        return np.array([_simulate(r, p) for r in rows]) - y
    best = None
    for start in ([math.log(.1), .1, 1., math.log(1.)],
                  [math.log(.3), 0., 2., math.log(2.)],
                  [math.log(.05), 1., 1., math.log(.3)]):
        res = least_squares(residual, np.array(start), max_nfev=250, x_scale='jac')
        if best is None or np.dot(res.fun, res.fun) < np.dot(best.fun, best.fun):
            best = res
    _PARAMS = (math.exp(best.x[0]), best.x[1], best.x[2], math.exp(best.x[3]))
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return _simulate(row, _PARAMS)