import math
import numpy as np
from scipy.optimize import least_squares

_P = None

def _z0(row, tau):
    if row.get('mem_conv') is not None:
        return float(row['mem_conv'])
    prefix = row.get('raw_prefix') or []
    if not prefix:
        return 0.52 * float(row['x_now'])
    z = 0.52 * float(prefix[0]['x'])
    tprev = float(prefix[0]['t'])
    xprev = float(prefix[0]['x'])
    for p in prefix[1:]:
        t = float(p['t']); x = float(p['x'])
        dt = max(0.0, t - tprev)
        e = math.exp(-dt / tau)
        # Exact response of the first-order kernel to linearly interpolated x.
        z = z * e + 0.52 * (xprev * (1.0 - e) + (x - xprev) / dt * (dt - tau * (1.0 - e)) if dt > 0 else 0.0)
        tprev, xprev = t, x
    return z

def _sim(row, pars):
    a, b, c, tau = map(float, pars)
    x = float(row['x_now']); z = _z0(row, tau); u = float(row['u_now'])
    H = max(0.0, float(row['H']))
    n = max(1, int(math.ceil(H / 0.004)))
    h = H / n
    def f(xv, zv):
        return -a*xv + b*zv + c*u, (0.52*xv-zv)/tau
    for _ in range(n):
        k1x,k1z=f(x,z)
        k2x,k2z=f(x+0.5*h*k1x,z+0.5*h*k1z)
        k3x,k3z=f(x+0.5*h*k2x,z+0.5*h*k2z)
        k4x,k4z=f(x+h*k3x,z+h*k3z)
        x += h*(k1x+2*k2x+2*k3x+k4x)/6.0
        z += h*(k1z+2*k2z+2*k3z+k4z)/6.0
    return x

def fit(train_rows):
    global _P
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    def resid(p):
        try:
            vals = np.array([_sim(r, p) for r in rows])
            if not np.all(np.isfinite(vals)):
                return np.full(len(rows), 1e6)
            return vals-y
        except (ValueError, OverflowError, ZeroDivisionError):
            return np.full(len(rows), 1e6)
    best = None
    starts = ([0.8,0.25,0.35,0.12], [1.2,0.5,0.5,0.3], [0.4,0.15,0.2,0.06])
    for p0 in starts:
        res = least_squares(resid, p0, bounds=([1e-4,0.0,0.0,0.005],[8.0,8.0,8.0,5.0]), max_nfev=3000)
        if best is None or np.dot(res.fun,res.fun) < np.dot(best.fun,best.fun):
            best = res
    _P = best.x
    return _P

def predict(row) -> float:
    if _P is None:
        raise RuntimeError('fit must be called before predict')
    return float(_sim(row, _P))