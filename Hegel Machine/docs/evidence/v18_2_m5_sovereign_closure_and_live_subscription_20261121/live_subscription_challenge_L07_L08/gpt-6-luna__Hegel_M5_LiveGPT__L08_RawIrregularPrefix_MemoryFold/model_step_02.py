import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _z0(row, tau):
    pref = row.get('raw_prefix')
    if row.get('mem_conv') is not None:
        return float(row['mem_conv'])
    if not pref:
        return 0.52 * float(row['x_now'])
    ts = [float(p['t']) for p in pref]
    xs = [float(p['x']) for p in pref]
    end = ts[-1]
    z = math.exp(-(end-ts[0])/tau) * 0.52 * xs[0]
    conv = 0.0
    for i in range(1, len(pref)):
        d = ts[i] - ts[i-1]
        conv += 0.5 * (math.exp(-(end-ts[i-1])/tau)*xs[i-1] + math.exp(-(end-ts[i])/tau)*xs[i]) * d / tau
    return z + 0.52 * conv

def _simulate(row, pars):
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
        k1 = f(q)
        k2 = f(q + 0.5*h*k1)
        k3 = f(q + 0.5*h*k2)
        k4 = f(q + h*k3)
        q += (h/6.0)*(k1+2*k2+2*k3+k4)
    return float(q[0])

def fit(train_rows):
    global _PARAMS
    def residual(p):
        return np.array([_simulate(r, p) - float(r['y']) for r in train_rows])
    starts = [np.array([0.3, 0.5, 0.5, 0.2]), np.array([0.5, 0.3, 0.8, 0.5]), np.array([0.2, 0.8, 0.4, 1.0])]
    best = None
    for x0 in starts:
        res = least_squares(residual, x0, bounds=([1e-5, 1e-5, 1e-5, 0.01], [5.0, 5.0, 5.0, 5.0]), max_nfev=3000)
        score = float(np.dot(res.fun, res.fun))
        if best is None or score < best[0]:
            best = (score, res.x)
    _PARAMS = best[1]
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    return _simulate(row, _PARAMS)
