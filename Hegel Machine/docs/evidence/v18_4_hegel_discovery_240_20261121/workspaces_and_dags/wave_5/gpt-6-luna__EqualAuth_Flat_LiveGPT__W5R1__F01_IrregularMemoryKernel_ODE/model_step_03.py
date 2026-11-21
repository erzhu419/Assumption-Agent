import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _z0(row, tau):
    if row.get('mem_conv') is not None:
        return float(row['mem_conv'])
    pts = row.get('raw_prefix') or []
    if not pts:
        return 0.52 * float(row['x_now'])
    ts = np.array([float(p['t']) for p in pts], dtype=float)
    xs = np.array([float(p['x']) for p in pts], dtype=float)
    order = np.argsort(ts)
    ts, xs = ts[order], xs[order]
    # Initialize at the oldest sample, then integrate tau*dz/dt=0.52*x-z
    z = 0.52 * xs[0]
    for i in range(1, len(ts)):
        dt = ts[i] - ts[i-1]
        if dt <= 0:
            continue
        decay = math.exp(-dt / tau)
        # Exact exponential update with linearly interpolated x across interval.
        slope = (xs[i] - xs[i-1]) / dt
        z = decay * z + 0.52 * (xs[i-1] * (1.0-decay) + slope * (dt - tau*(1.0-decay)))
    return float(z)

def _simulate(row, p):
    a, b, c, tau = [float(v) for v in p]
    x = float(row['x_now'])
    z = _z0(row, tau)
    u = float(row['u_now'])
    H = float(row['H'])
    n = max(1, int(math.ceil(H / 0.004)))
    dt = H / n
    def f(v):
        xx, zz = v
        return np.array([-a*xx + b*zz + c*u, (0.52*xx-zz)/tau], dtype=float)
    v = np.array([x, z], dtype=float)
    for _ in range(n):
        k1=f(v); k2=f(v+0.5*dt*k1); k3=f(v+0.5*dt*k2); k4=f(v+dt*k3)
        v = v + dt*(k1+2*k2+2*k3+k4)/6.0
    return float(v[0])

def fit(train_rows):
    global _PARAMS
    def residual(p):
        return np.array([_simulate(r,p)-float(r['y']) for r in train_rows], dtype=float)
    starts = [[0.4,0.6,0.8,0.5],[0.8,0.4,1.0,1.0],[0.2,1.0,0.5,2.0]]
    best = None
    for s in starts:
        res = least_squares(residual, np.array(s,dtype=float), bounds=([0.001,0.001,0.001,0.02],[5.0,5.0,5.0,10.0]), max_nfev=3000)
        if best is None or np.sum(res.fun**2) < np.sum(best.fun**2):
            best = res
    _PARAMS = best.x.copy()
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_simulate(row, _PARAMS))