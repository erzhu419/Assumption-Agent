import math
import numpy as np
from scipy.optimize import least_squares

_P = None

def _z0(row, tau):
    pref = row.get('raw_prefix') or []
    if not pref:
        return float(row.get('mem_conv') or 0.52 * row['x_now'])
    ts = np.array([p['t'] for p in pref], dtype=float)
    xs = np.array([p['x'] for p in pref], dtype=float)
    end = ts[-1]
    z = math.exp(-(end-ts[0])/tau) * 0.52 * xs[0]
    for i in range(1, len(ts)):
        dt = ts[i]-ts[i-1]
        z += 0.52 * 0.5 * (math.exp(-(end-ts[i-1])/tau)*xs[i-1] + math.exp(-(end-ts[i])/tau)*xs[i]) * dt / tau
    return z

def _sim(row, p):
    a, b, c, tau = p
    x = float(row['x_now'])
    z = float(row['mem_conv']) if row.get('mem_conv') is not None else _z0(row, tau)
    u = float(row['u_now'])
    H = float(row['H'])
    n = max(1, int(math.ceil(H/0.004)))
    dt = H/n
    def f(q):
        xx, zz = q
        return np.array([-a*xx+b*zz+c*u, (0.52*xx-zz)/tau])
    q = np.array([x,z], dtype=float)
    for _ in range(n):
        k1=f(q); k2=f(q+0.5*dt*k1); k3=f(q+0.5*dt*k2); k4=f(q+dt*k3)
        q += dt*(k1+2*k2+2*k3+k4)/6.0
    return float(q[0])

def fit(train_rows):
    global _P
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        return np.array([_sim(r,p) for r in train_rows])-y
    result = least_squares(residual, x0=np.array([0.8, 0.25, 0.8, 0.22]), bounds=([0.001,0.001,0.001,0.01],[5.0,5.0,5.0,3.0]), max_nfev=300)
    _P = result.x
    return _P

def predict(row):
    if _P is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return _sim(row, _P)
