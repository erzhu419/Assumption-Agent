import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _z0(row, tau):
    pref = row.get('raw_prefix') or []
    if not pref:
        return 0.52 * float(row.get('x_now', 0.0))
    ts = np.array([float(q['t']) for q in pref], dtype=float)
    xs = np.array([float(q['x']) for q in pref], dtype=float)
    end = ts[-1]
    z = 0.52 * xs[0] * math.exp(-(end-ts[0])/tau)
    for k in range(1, len(ts)):
        dt = ts[k]-ts[k-1]
        z += 0.52 * 0.5 * (math.exp(-(end-ts[k-1])/tau)*xs[k-1] + math.exp(-(end-ts[k])/tau)*xs[k]) * dt / tau
    return z

def _sim(row, p):
    a, b, c, tau = [float(v) for v in p]
    x = float(row['x_now'])
    z = float(row['mem_conv']) if row.get('mem_conv') is not None else _z0(row, tau)
    u = float(row['u_now'])
    H = float(row['H'])
    n = max(1, int(math.ceil(H/0.004)))
    dt = H/n
    def f(xv, zv):
        return -a*xv + b*zv + c*u, (0.52*xv-zv)/tau
    for _ in range(n):
        k1x,k1z=f(x,z)
        k2x,k2z=f(x+0.5*dt*k1x,z+0.5*dt*k1z)
        k3x,k3z=f(x+0.5*dt*k2x,z+0.5*dt*k2z)
        k4x,k4z=f(x+dt*k3x,z+dt*k3z)
        x += dt*(k1x+2*k2x+2*k3x+k4x)/6
        z += dt*(k1z+2*k2z+2*k3z+k4z)/6
    return x

def fit(train_rows):
    global _PARAMS
    rows = [r for r in train_rows if isinstance(r, dict) and 'y' in r]
    if len(rows) < 4:
        raise ValueError('Need labeled training rows with y')
    def residual(p):
        return np.array([_sim(r,p)-float(r['y']) for r in rows])
    res = least_squares(residual, np.array([0.5, 0.3, 0.5, 0.25]), bounds=([0.001,-3,-3,0.01],[5,3,3,5]), max_nfev=3000)
    _PARAMS = res.x.copy()
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise ValueError('fit must be called before predict')
    return float(_sim(row, _PARAMS))