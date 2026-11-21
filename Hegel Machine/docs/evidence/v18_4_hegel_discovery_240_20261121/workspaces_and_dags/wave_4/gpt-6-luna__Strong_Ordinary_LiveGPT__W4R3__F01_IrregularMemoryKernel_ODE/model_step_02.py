import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = np.array([0.5, 0.3, 0.5, 0.15], dtype=float)

def _z0(row, tau):
    if row.get('mem_conv') is not None:
        return float(row['mem_conv'])
    prefix = row.get('raw_prefix') or []
    if not prefix:
        return 0.52 * float(row['x_now'])
    ts = np.array([float(p['t']) for p in prefix], dtype=float)
    xs = np.array([float(p['x']) for p in prefix], dtype=float)
    order = np.argsort(ts)
    ts, xs = ts[order], xs[order]
    z = 0.52 * xs[0]
    for i in range(1, len(ts)):
        dt = ts[i] - ts[i-1]
        if dt <= 0:
            continue
        decay = math.exp(-dt / tau)
        # Exact exponential weighting with linearly interpolated x on this interval.
        slope = (xs[i] - xs[i-1]) / dt
        z = decay * z + 0.52 * (xs[i-1] * (1.0-decay) + slope * (dt - tau*(1.0-decay)))
    return z

def _simulate(row, p):
    a, b, c, tau = [float(v) for v in p]
    x = float(row['x_now'])
    z = _z0(row, tau)
    u = float(row['u_now'])
    H = max(0.0, float(row['H']))
    n = max(1, int(math.ceil(H / 0.004)))
    dt = H / n
    def f(xv, zv):
        return -a*xv + b*zv + c*u, (0.52*xv-zv)/tau
    for _ in range(n):
        k1x,k1z=f(x,z)
        k2x,k2z=f(x+0.5*dt*k1x,z+0.5*dt*k1z)
        k3x,k3z=f(x+0.5*dt*k2x,z+0.5*dt*k2z)
        k4x,k4z=f(x+dt*k3x,z+dt*k3z)
        x += dt*(k1x+2*k2x+2*k3x+k4x)/6.0
        z += dt*(k1z+2*k2z+2*k3z+k4z)/6.0
    return x

def fit(train_rows):
    global _PARAMS
    ys = np.array([float(r['y']) for r in train_rows])
    def residual(logp):
        p = np.exp(logp)
        return np.array([_simulate(r, p) for r in train_rows]) - ys
    starts = [np.log([0.5,0.3,0.5,0.15]), np.log([0.8,0.5,0.8,0.3]), np.log([0.3,0.8,0.4,0.08])]
    best = None
    for s in starts:
        res = least_squares(residual, s, max_nfev=250)
        if best is None or np.sum(res.fun**2) < np.sum(best.fun**2):
            best = res
    _PARAMS = np.exp(best.x)
    return _PARAMS

def predict(row) -> float:
    return float(_simulate(row, _PARAMS))