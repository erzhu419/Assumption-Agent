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
    d = ts[-1] - ts[0]
    conv = 0.0
    for i in range(1, len(ts)):
        a, b = ts[-1] - ts[i-1], ts[-1] - ts[i]
        conv += 0.5 * (math.exp(-a/tau)*xs[i-1] + math.exp(-b/tau)*xs[i]) * (ts[i]-ts[i-1]) / tau
    return math.exp(-d/tau) * 0.52 * xs[0] + 0.52 * conv

def _sim(row, p):
    a, b, c, tau = p
    x, z = float(row['x_now']), _z0(row, tau)
    u, H = float(row['u_now']), float(row['H'])
    n = max(1, int(math.ceil(H/0.004)))
    h = H/n
    def f(q):
        xx, zz = q
        return np.array([-a*xx + b*zz + c*u, (0.52*xx-zz)/tau])
    for _ in range(n):
        q = np.array([x,z])
        k1=f(q); k2=f(q+h*k1/2); k3=f(q+h*k2/2); k4=f(q+h*k3)
        q += h*(k1+2*k2+2*k3+k4)/6
        x,z = q
    return float(x)

def fit(train_rows):
    global _P
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        return np.array([_sim(r,p) for r in train_rows])-y
    result = least_squares(residual, [0.5, 0.3, 0.7, 0.2], bounds=([0.001,0.001,0.001,0.005],[5,5,5,5]), max_nfev=3000)
    _P = result.x
    return _P.copy()

def predict(row):
    if _P is None:
        raise ValueError('fit(train_rows) must be called first')
    return _sim(row, _P)
