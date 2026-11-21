import math
import numpy as np
from scipy.optimize import least_squares

def _z0(row, tau):
    if row.get('mem_conv') is not None:
        return float(row['mem_conv'])
    pts = row.get('raw_prefix') or []
    if not pts:
        return 0.52 * float(row['x_now'])
    ts = np.array([float(p['t']) for p in pts], dtype=float)
    xs = np.array([float(p['x']) for p in pts], dtype=float)
    t1 = ts[-1]
    z = 0.52 * xs[0] * math.exp(-(t1-ts[0])/tau)
    for i in range(1, len(ts)):
        dt = ts[i] - ts[i-1]
        w = math.exp(-(t1-ts[i])/tau)
        w0 = math.exp(-(t1-ts[i-1])/tau)
        z += 0.52 * 0.5 * (xs[i-1]*w0 + xs[i]*w) * dt / tau
    return z

def _sim(row, p):
    a, b, c, tau = p
    tau = max(float(tau), 1e-5)
    x = float(row['x_now'])
    z = _z0(row, tau)
    u = float(row['u_now'])
    H = max(0.0, float(row['H']))
    n = max(1, int(math.ceil(H / 0.004)))
    dt = H / n
    def f(xx, zz):
        return -a*xx + b*zz + c*u, (0.52*xx-zz)/tau
    for _ in range(n):
        k1x,k1z=f(x,z)
        k2x,k2z=f(x+0.5*dt*k1x,z+0.5*dt*k1z)
        k3x,k3z=f(x+0.5*dt*k2x,z+0.5*dt*k2z)
        k4x,k4z=f(x+dt*k3x,z+dt*k3z)
        x += dt*(k1x+2*k2x+2*k3x+k4x)/6
        z += dt*(k1z+2*k2z+2*k3z+k4z)/6
    return x

def fit(train_rows):
    ys = np.array([float(r['y']) for r in train_rows])
    def resid(p):
        return np.array([_sim(r,p) for r in train_rows]) - ys
    best = None
    for p0 in ([0.4,0.5,0.5,0.2], [0.8,0.3,0.8,0.5], [0.2,0.8,0.3,1.0]):
        res = least_squares(resid, p0, bounds=([0,0,-5,0.005],[5,5,5,5]), max_nfev=3000)
        if best is None or np.sum(res.fun**2) < np.sum(best.fun**2):
            best = res
    return best.x.tolist()

def predict(row):
    return float(_sim(row, _MODEL_PARAMS))
