import math
import numpy as np
from scipy.optimize import least_squares

def _z0(row, tau):
    pts = row.get('raw_prefix') or []
    if not pts:
        return 0.52 * float(row['x_now'])
    t0 = float(pts[0]['t'])
    z = 0.52 * float(pts[0]['x'])
    for p, q in zip(pts[:-1], pts[1:]):
        dt = float(q['t']) - float(p['t'])
        xm = 0.5 * (float(p['x']) + float(q['x']))
        e = math.exp(-dt / tau)
        z = 0.52 * xm + (z - 0.52 * xm) * e
    return z

def _integrate(row, pars):
    a, b, c, tau = [float(v) for v in pars]
    x = float(row['x_now'])
    z = float(row['mem_conv']) if row.get('mem_conv') is not None else _z0(row, tau)
    u = float(row['u_now'])
    H = float(row['H'])
    n = max(1, int(math.ceil(H / 0.004)))
    dt = H / n
    def f(xv, zv):
        return -a*xv + b*zv + c*u, (0.52*xv-zv)/tau
    for _ in range(n):
        k1x,k1z=f(x,z)
        k2x,k2z=f(x+dt*k1x/2,z+dt*k1z/2)
        k3x,k3z=f(x+dt*k2x/2,z+dt*k2z/2)
        k4x,k4z=f(x+dt*k3x,z+dt*k3z)
        x += dt*(k1x+2*k2x+2*k3x+k4x)/6
        z += dt*(k1z+2*k2z+2*k3z+k4z)/6
    return x

def fit(train_rows):
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_integrate(r,p) for r in train_rows]) - y
    best = None
    for p0 in ([0.3,0.2,0.5,0.2],[0.8,0.1,0.8,0.5],[0.1,0.5,0.2,0.1]):
        sol = least_squares(residual, p0, bounds=([1e-5,0,0,0.005],[5,5,5,5]), max_nfev=3000)
        if best is None or np.sum(sol.fun**2) < np.sum(best.fun**2): best = sol
    return best.x.tolist()

def predict(row):
    return float(_integrate(row, _PARAMS))