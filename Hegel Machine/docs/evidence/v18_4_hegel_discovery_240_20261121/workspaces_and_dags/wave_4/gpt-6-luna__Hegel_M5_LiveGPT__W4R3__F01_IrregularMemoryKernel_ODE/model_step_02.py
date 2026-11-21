import math
import numpy as np
from scipy.optimize import least_squares

def _z0(row, tau):
    if row.get('mem_conv') is not None:
        return float(row['mem_conv'])
    pref = row.get('raw_prefix') or []
    if not pref:
        return 0.52 * float(row['x_now'])
    ts = np.array([float(p['t']) for p in pref], dtype=float)
    xs = np.array([float(p['x']) for p in pref], dtype=float)
    z = 0.52 * xs[0] * math.exp(-(ts[-1] - ts[0]) / tau)
    conv = 0.0
    for k in range(1, len(pref)):
        d = ts[k] - ts[k-1]
        conv += 0.5 * (math.exp(-(ts[-1]-ts[k-1])/tau)*xs[k-1] + math.exp(-(ts[-1]-ts[k])/tau)*xs[k]) * d / tau
    return z + 0.52 * conv

def _predict_params(row, p):
    a, b, c, tau = map(float, p)
    x = float(row['x_now'])
    z = _z0(row, tau)
    u = float(row['u_now'])
    H = float(row['H'])
    n = max(1, int(math.ceil(H / 0.004)))
    dt = H / n
    def f(x, z):
        return -a*x + b*z + c*u, (0.52*x-z)/tau
    for _ in range(n):
        k1x,k1z = f(x,z)
        k2x,k2z = f(x+0.5*dt*k1x,z+0.5*dt*k1z)
        k3x,k3z = f(x+0.5*dt*k2x,z+0.5*dt*k2z)
        k4x,k4z = f(x+dt*k3x,z+dt*k3z)
        x += dt*(k1x+2*k2x+2*k3x+k4x)/6.0
        z += dt*(k1z+2*k2z+2*k3z+k4z)/6.0
    return x

def fit(train_rows):
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_predict_params(r,p) for r in train_rows]) - y
    starts = ([0.5,0.5,0.5,0.2], [1.0,0.3,0.8,0.5], [0.2,1.0,0.3,1.0])
    best = None
    for p0 in starts:
        res = least_squares(residual, p0, bounds=([0.001,0.001,0.001,0.01],[5.0,5.0,5.0,5.0]), max_nfev=3000)
        if best is None or np.sum(res.fun**2) < np.sum(best.fun**2):
            best = res
    return best.x

def predict(row):
    return float(_predict_params(row, fit([dict(row, y=0.0)])))