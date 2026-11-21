import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _initial_memory(row, tau):
    prefix = row.get('raw_prefix') or []
    if not prefix:
        return 0.52 * float(row['x_now'])
    prefix = sorted(prefix, key=lambda p: p['t'])
    z = 0.52 * float(prefix[0]['x'])
    for p0, p1 in zip(prefix[:-1], prefix[1:]):
        dt = float(p1['t']) - float(p0['t'])
        if dt <= 0:
            continue
        decay = math.exp(-dt / tau)
        # Trapezoidal forcing over this irregular interval.
        xmid = 0.5 * (float(p0['x']) + float(p1['x']))
        z = z * decay + 0.52 * xmid * (1.0 - decay)
    return z

def _integrate(row, pars):
    a, b, c, tau = [float(v) for v in pars]
    x = float(row['x_now'])
    z = float(row['mem_conv']) if row.get('mem_conv') is not None else _initial_memory(row, tau)
    u = float(row['u_now'])
    H = float(row['H'])
    n = max(1, int(math.ceil(H / 0.004)))
    dt = H / n
    def deriv(state):
        xx, zz = state
        return np.array([-a * xx + b * zz + c * u, (0.52 * xx - zz) / tau], dtype=float)
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
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([_integrate(r, p) for r in rows]) - y
    starts = [np.array([0.5, 0.3, 0.5, 0.2]), np.array([1.0, 0.5, 1.0, 0.5]), np.array([0.2, 0.1, 0.8, 0.05])]
    best = None
    for p0 in starts:
        result = least_squares(residual, p0, bounds=([1e-5, 0.0, 0.0, 1e-4], [10.0, 10.0, 10.0, 10.0]), max_nfev=3000)
        if best is None or np.sum(result.fun**2) < np.sum(best.fun**2):
            best = result
    _PARAMS = best.x
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_integrate(row, _PARAMS))