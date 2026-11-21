import math
import numpy as np
from scipy.optimize import least_squares

def _z0(row):
    if row.get('mem_conv') is not None:
        return float(row['mem_conv'])
    pts = row.get('raw_prefix') or []
    if not pts:
        return 0.52 * float(row['x_now'])
    z = 0.52 * float(pts[0]['x'])
    t0 = float(pts[0]['t'])
    tl = float(pts[-1]['t'])
    tau = _PARAM_TAU
    for p, q in zip(pts[:-1], pts[1:]):
        dt = float(q['t']) - float(p['t'])
        if dt <= 0:
            continue
        target = 0.52 * 0.5 * (float(p['x']) + float(q['x']))
        z = target + (z - target) * math.exp(-dt / tau)
    return z

def _sim(row, pars):
    a, b, c, tau = pars
    global _PARAM_TAU
    _PARAM_TAU = tau
    x = float(row['x_now'])
    z = _z0(row)
    u = float(row['u_now'])
    H = float(row['H'])
    n = max(1, int(math.ceil(H / 0.004)))
    dt = H / n
    def f(x, z):
        return -a*x + b*z + c*u, (0.52*x-z)/tau
    for _ in range(n):
        k1x,k1z=f(x,z)
        k2x,k2z=f(x+dt*k1x/2,z+dt*k1z/2)
        k3x,k3z=f(x+dt*k2x/2,z+dt*k2z/2)
        k4x,k4z=f(x+dt*k3x,z+dt*k3z)
        x += dt*(k1x+2*k2x+2*k3x+k4x)/6
        z += dt*(k1z+2*k2z+2*k3z+k4z)/6
    return x

def fit(train_rows):
    global _PARAM_TAU
    def residual(p):
        _PARAM_TAU = float(p[3])
        return np.array([_sim(r,p)-float(r['y']) for r in train_rows])
    res = least_squares(residual, [0.5,0.2,0.5,0.2], bounds=([0,0,0,0.005],[10,10,10,10]), max_nfev=5000)
    params = res.x.copy()
    def predict(row):
        return float(_sim(row,params))
    return predict

def predict(row):
    return float('nan')