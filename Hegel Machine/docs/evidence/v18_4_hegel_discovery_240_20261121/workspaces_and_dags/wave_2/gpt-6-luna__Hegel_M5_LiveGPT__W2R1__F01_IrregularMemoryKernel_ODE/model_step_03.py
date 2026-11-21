import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _z0(row, tau):
    if row.get('mem_conv') is not None:
        return float(row['mem_conv'])
    pref = row.get('raw_prefix') or []
    if not pref:
        return 0.52 * float(row['x_now'])
    ts = np.array([float(p['t']) for p in pref], dtype=float)
    xs = np.array([float(p['x']) for p in pref], dtype=float)
    end = ts[-1]
    z = math.exp(-(end - ts[0]) / tau) * 0.52 * xs[0]
    for k in range(1, len(pref)):
        dt = ts[k] - ts[k-1]
        q0 = math.exp(-(end - ts[k-1]) / tau) * xs[k-1]
        q1 = math.exp(-(end - ts[k]) / tau) * xs[k]
        z += 0.52 * 0.5 * (q0 + q1) * dt / tau
    return z

def _sim(row, pars):
    a, b, c, tau = [float(v) for v in pars]
    x = float(row['x_now'])
    z = _z0(row, tau)
    u = float(row['u_now'])
    H = float(row['H'])
    n = max(1, int(math.ceil(H / 0.004)))
    h = H / n
    def f(state):
        xx, zz = state
        return np.array([-a*xx + b*zz + c*u, (0.52*xx - zz)/tau])
    state = np.array([x, z], dtype=float)
    for _ in range(n):
        k1 = f(state)
        k2 = f(state + 0.5*h*k1)
        k3 = f(state + 0.5*h*k2)
        k4 = f(state + h*k3)
        state += (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    return float(state[0])

def fit(train_rows):
    global _PARAMS
    def residual(p):
        return np.array([_sim(r, p) - float(r['y']) for r in train_rows])
    result = least_squares(residual, x0=np.array([0.8, 0.3, 0.5, 0.15]), bounds=([0.001, 0.001, 0.001, 0.005], [5.0, 5.0, 5.0, 5.0]), max_nfev=3000)
    _PARAMS = result.x.copy()
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return _sim(row, _PARAMS)
