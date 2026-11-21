import math
import numpy as np
from scipy.optimize import least_squares

def _z0(row, tau):
    pref = row.get('raw_prefix') or []
    if row.get('mem_conv') is not None:
        return float(row['mem_conv'])
    if not pref:
        return 0.52 * float(row['x_now'])
    ts = np.array([float(q['t']) for q in pref], dtype=float)
    xs = np.array([float(q['x']) for q in pref], dtype=float)
    end = ts[-1]
    z = 0.52 * xs[0] * math.exp(-(end-ts[0])/tau)
    for i in range(1, len(ts)):
        dt = ts[i] - ts[i-1]
        z += 0.52 * 0.5 * (math.exp(-(end-ts[i-1])/tau)*xs[i-1] + math.exp(-(end-ts[i])/tau)*xs[i]) * dt / tau
    return z

def _sim(row, p):
    a, b, c, tau = [float(v) for v in p]
    x = float(row['x_now'])
    z = _z0(row, tau)
    H = float(row['H'])
    u = float(row['u_now'])
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
    rows = [r for r in train_rows if 'y' in r]
    if not rows:
        raise ValueError('Training rows must include y targets')
    yy = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([_sim(r,p) for r in rows]) - yy
    return least_squares(residual, x0=np.array([0.4,0.2,0.5,0.3]), bounds=([0.001,0.001,0.001,0.01],[5,5,5,5]), max_nfev=3000).x

def predict(row):
    return float(_sim(row, fit_params))