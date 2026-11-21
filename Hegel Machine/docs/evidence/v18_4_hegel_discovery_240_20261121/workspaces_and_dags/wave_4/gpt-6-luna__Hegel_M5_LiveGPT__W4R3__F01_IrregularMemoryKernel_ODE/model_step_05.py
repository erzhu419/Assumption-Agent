import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _z0(row, tau):
    pref = row.get('raw_prefix') or []
    if not pref:
        return float(row.get('mem_conv') or (0.52 * row['x_now']))
    ts = np.array([float(p['t']) for p in pref], dtype=float)
    xs = np.array([float(p['x']) for p in pref], dtype=float)
    end = ts[-1]
    z = math.exp(-(end-ts[0])/tau) * 0.52 * xs[0]
    for i in range(1, len(ts)):
        dt = ts[i] - ts[i-1]
        z += 0.52 * 0.5 * (math.exp(-(end-ts[i-1])/tau)*xs[i-1] + math.exp(-(end-ts[i])/tau)*xs[i]) * dt / tau
    return z

def _predict(row, p):
    a, b, c, tau = map(float, p)
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
        k2x,k2z=f(x+0.5*dt*k1x,z+0.5*dt*k1z)
        k3x,k3z=f(x+0.5*dt*k2x,z+0.5*dt*k2z)
        k4x,k4z=f(x+dt*k3x,z+dt*k3z)
        x += dt*(k1x+2*k2x+2*k3x+k4x)/6.0
        z += dt*(k1z+2*k2z+2*k3z+k4z)/6.0
    return x

def fit(train_rows):
    global _PARAMS
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_predict(r,p) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([0.3, 0.2, 0.8, 0.15]), bounds=([0.001, -3.0, -3.0, 0.005], [4.0, 3.0, 3.0, 3.0]), max_nfev=3000)
    _PARAMS = result.x
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit must be called first')
    return float(_predict(row, _PARAMS))