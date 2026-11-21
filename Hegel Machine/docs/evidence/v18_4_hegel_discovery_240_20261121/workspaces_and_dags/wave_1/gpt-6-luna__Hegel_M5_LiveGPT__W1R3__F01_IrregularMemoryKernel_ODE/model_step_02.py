import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _initial_memory(row, tau):
    if row.get('mem_conv') is not None:
        return float(row['mem_conv'])
    pref = row.get('raw_prefix') or []
    if not pref:
        return 0.52 * float(row['x_now'])
    ts = np.asarray([p['t'] for p in pref], dtype=float)
    xs = np.asarray([p['x'] for p in pref], dtype=float)
    tend = ts[-1]
    total = math.exp(-(tend - ts[0]) / tau) * 0.52 * xs[0]
    for k in range(1, len(ts)):
        dt = ts[k] - ts[k-1]
        q0 = math.exp(-(tend - ts[k-1]) / tau) * xs[k-1]
        q1 = math.exp(-(tend - ts[k]) / tau) * xs[k]
        total += 0.52 * 0.5 * (q0 + q1) * dt / tau
    return total

def _simulate(row, pars):
    a, b, c, tau = pars
    x = float(row['x_now'])
    z = _initial_memory(row, tau)
    u = float(row['u_now'])
    H = float(row['H'])
    n = max(1, int(math.ceil(H / 0.004)))
    h = H / n
    def f(x, z):
        return -a*x + b*z + c*u, (0.52*x-z)/tau
    for _ in range(n):
        k1x,k1z = f(x,z)
        k2x,k2z = f(x+0.5*h*k1x,z+0.5*h*k1z)
        k3x,k3z = f(x+0.5*h*k2x,z+0.5*h*k2z)
        k4x,k4z = f(x+h*k3x,z+h*k3z)
        x += h*(k1x+2*k2x+2*k3x+k4x)/6.0
        z += h*(k1z+2*k2z+2*k3z+k4z)/6.0
    return x

def fit(train_rows):
    global _PARAMS
    def residual(p):
        return np.asarray([_simulate(r, p)-float(r['y']) for r in train_rows])
    result = least_squares(residual, x0=np.array([0.8, 0.5, 0.8, 0.18]),
                           bounds=([0.001, 0.001, 0.001, 0.005], [5.0, 5.0, 5.0, 5.0]),
                           max_nfev=3000)
    _PARAMS = result.x
    return _PARAMS.copy()

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return float(_simulate(row, _PARAMS))