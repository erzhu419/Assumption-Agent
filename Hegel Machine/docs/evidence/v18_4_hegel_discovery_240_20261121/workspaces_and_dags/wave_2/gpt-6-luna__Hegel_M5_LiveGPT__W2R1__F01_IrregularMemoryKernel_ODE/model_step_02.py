import math
import numpy as np
from scipy.optimize import least_squares

def _z0(row, tau):
    if row.get('mem_conv') is not None:
        return float(row['mem_conv'])
    pref = row.get('raw_prefix')
    if not pref:
        return 0.52 * float(row['x_now'])
    ts = np.array([p['t'] for p in pref], dtype=float)
    xs = np.array([p['x'] for p in pref], dtype=float)
    end = ts[-1]
    z = math.exp(-(end-ts[0])/tau) * 0.52 * xs[0]
    conv = 0.0
    for k in range(1, len(ts)):
        dt = ts[k]-ts[k-1]
        conv += 0.5 * (math.exp(-(end-ts[k-1])/tau)*xs[k-1] + math.exp(-(end-ts[k])/tau)*xs[k]) * dt / tau
    return z + 0.52 * conv

def _sim(row, pars):
    a, b, c, tau = pars
    x = float(row['x_now'])
    z = _z0(row, tau)
    u = float(row['u_now'])
    H = float(row['H'])
    n = max(1, int(math.ceil(H / 0.004)))
    h = H / n
    def f(q):
        xx, zz = q
        return np.array([-a*xx + b*zz + c*u, (0.52*xx-zz)/tau])
    q = np.array([x, z], dtype=float)
    for _ in range(n):
        k1 = f(q); k2 = f(q + 0.5*h*k1); k3 = f(q + 0.5*h*k2); k4 = f(q + h*k3)
        q += (h/6.0)*(k1+2*k2+2*k3+k4)
    return float(q[0])

def fit(train_rows):
    def residual(p):
        return np.array([_sim(r, p)-float(r['y']) for r in train_rows])
    starts = [np.array([0.4, 0.2, 0.5, 0.3]), np.array([0.8, 0.4, 0.8, 0.8]), np.array([0.2, 0.1, 0.4, 1.5])]
    sols = [least_squares(residual, s, bounds=([1e-5, 1e-5, 1e-5, 0.02], [5.0, 5.0, 5.0, 10.0]), max_nfev=3000) for s in starts]
    return min(sols, key=lambda q: float(np.dot(q.fun, q.fun))).x

def predict(row):
    return _sim(row, fit([row]) if False else _PARAMS)
