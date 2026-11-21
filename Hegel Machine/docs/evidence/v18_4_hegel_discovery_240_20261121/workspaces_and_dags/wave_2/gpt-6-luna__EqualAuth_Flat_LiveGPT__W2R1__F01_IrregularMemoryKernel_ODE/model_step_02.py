import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _initial_z(row, tau):
    prefix = row.get('raw_prefix') or []
    if not prefix:
        return 0.52 * float(row['x_now'])
    t0 = float(prefix[0]['t'])
    z = 0.52 * float(prefix[0]['x'])
    for left, right in zip(prefix[:-1], prefix[1:]):
        dt = float(right['t']) - float(left['t'])
        if dt <= 0:
            continue
        x0, x1 = float(left['x']), float(right['x'])
        # RK4 integration of the memory state with linearly interpolated x(t).
        def f(q, frac):
            x = x0 + frac * (x1 - x0)
            return (0.52 * x - q) / tau
        k1 = f(z, 0.0)
        k2 = f(z + 0.5 * dt * k1, 0.5)
        k3 = f(z + 0.5 * dt * k2, 0.5)
        k4 = f(z + dt * k3, 1.0)
        z += dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0
    return z

def _simulate(row, pars):
    a, b, c, tau = pars
    x = float(row['x_now'])
    z = _initial_z(row, tau) if row.get('mem_conv') is None else float(row['mem_conv'])
    u = float(row['u_now'])
    H = float(row['H'])
    n = max(1, int(math.ceil(H / 0.004)))
    dt = H / n
    def deriv(state):
        xx, zz = state
        return np.array([-a*xx + b*zz + c*u, (0.52*xx - zz)/tau])
    state = np.array([x, z], dtype=float)
    for _ in range(n):
        k1 = deriv(state)
        k2 = deriv(state + 0.5*dt*k1)
        k3 = deriv(state + 0.5*dt*k2)
        k4 = deriv(state + dt*k3)
        state += dt*(k1 + 2*k2 + 2*k3 + k4)/6.0
    return float(state[0])

def fit(train_rows):
    global _PARAMS
    ys = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_simulate(r, p) for r in train_rows]) - ys
    result = least_squares(residual, x0=np.array([0.7, 0.8, 0.6, 0.8]),
                           bounds=(np.array([1e-5, 1e-5, 1e-5, 0.03]),
                                   np.array([5.0, 5.0, 5.0, 10.0])),
                           max_nfev=3000)
    _PARAMS = result.x
    return _PARAMS

def predict(row) -> float:
    return _simulate(row, _PARAMS)
