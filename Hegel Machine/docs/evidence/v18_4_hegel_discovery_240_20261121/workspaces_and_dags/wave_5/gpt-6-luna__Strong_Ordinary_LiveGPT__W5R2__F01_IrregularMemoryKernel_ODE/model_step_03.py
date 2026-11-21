import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _initial_memory(row, tau):
    prefix = row.get('raw_prefix') or []
    if not prefix:
        return float(row.get('mem_conv') if row.get('mem_conv') is not None else 0.52 * row['x_now'])
    z = 0.52 * float(prefix[0]['x'])
    t_latest = float(prefix[-1]['t'])
    for left, right in zip(prefix[:-1], prefix[1:]):
        t0, t1 = float(left['t']), float(right['t'])
        dt = t1 - t0
        if dt <= 0:
            continue
        x0, x1 = float(left['x']), float(right['x'])
        e = math.exp(-dt / tau)
        slope = (x1 - x0) / dt
        z = e*z + 0.52*(x0*(1.0-e) + slope*(dt - tau*(1.0-e)))
    return z

def _simulate(row, p):
    a, b, c, tau = map(float, p)
    x = float(row['x_now'])
    z = _initial_memory(row, tau) if row.get('mem_conv') is None else float(row['mem_conv'])
    u = float(row['u_now'])
    H = float(row['H'])
    n = max(1, int(math.ceil(H / 0.004)))
    dt = H / n
    def f(xv, zv):
        return -a*xv + b*zv + c*u, (0.52*xv-zv)/tau
    for _ in range(n):
        k1x,k1z = f(x,z)
        k2x,k2z = f(x+0.5*dt*k1x,z+0.5*dt*k1z)
        k3x,k3z = f(x+0.5*dt*k2x,z+0.5*dt*k2z)
        k4x,k4z = f(x+dt*k3x,z+dt*k3z)
        x += dt*(k1x+2*k2x+2*k3x+k4x)/6.0
        z += dt*(k1z+2*k2z+2*k3z+k4z)/6.0
    return x

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    y = np.asarray([r['y'] for r in rows], dtype=float)
    def residual(p):
        return np.asarray([_simulate(r,p) for r in rows]) - y
    result = least_squares(residual, x0=[0.8, 0.5, 0.8, 0.15], bounds=([0.001,0.001,0.001,0.005],[8.0,8.0,8.0,3.0]), max_nfev=3000)
    _PARAMS = result.x
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_simulate(row, _PARAMS))