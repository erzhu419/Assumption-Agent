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
    end = ts[-1]
    z = math.exp(-(end-ts[0])/tau) * 0.52 * xs[0]
    for k in range(1, len(ts)):
        dt = ts[k]-ts[k-1]
        z += 0.52 * 0.5 * (math.exp(-(end-ts[k-1])/tau)*xs[k-1] + math.exp(-(end-ts[k])/tau)*xs[k]) * dt / tau
    return z

def _predict_params(row, p):
    a, b, c, tau = map(float, p)
    x = float(row['x_now'])
    z = _z0(row, tau)
    u = float(row['u_now'])
    H = float(row['H'])
    n = max(1, int(math.ceil(H/0.004)))
    dt = H/n
    def f(xv, zv):
        return -a*xv+b*zv+c*u, (0.52*xv-zv)/tau
    for _ in range(n):
        k1x,k1z=f(x,z)
        k2x,k2z=f(x+dt*k1x/2,z+dt*k1z/2)
        k3x,k3z=f(x+dt*k2x/2,z+dt*k2z/2)
        k4x,k4z=f(x+dt*k3x,z+dt*k3z)
        x += dt*(k1x+2*k2x+2*k3x+k4x)/6
        z += dt*(k1z+2*k2z+2*k3z+k4z)/6
    return x

def fit(train_rows):
    def residual(p):
        return np.array([_predict_params(r,p)-float(r['y']) for r in train_rows])
    result = least_squares(residual, x0=np.array([0.8,0.5,0.5,0.25]), bounds=([0.001,0.001,0.001,0.01],[5.0,5.0,5.0,3.0]), max_nfev=3000)
    return result.x

def predict(row):
    p = fit(_TRAIN_ROWS)
    return float(_predict_params(row,p))