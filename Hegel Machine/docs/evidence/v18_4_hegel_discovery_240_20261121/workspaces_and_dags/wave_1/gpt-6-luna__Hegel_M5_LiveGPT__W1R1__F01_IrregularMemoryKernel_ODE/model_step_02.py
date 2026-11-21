import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _z0(row, tau):
    if row.get('mem_conv') is not None:
        return float(row['mem_conv'])
    pref = row.get('raw_prefix') or []
    if not pref:
        return 0.52 * float(row['x_now'])
    ts = np.asarray([float(p['t']) for p in pref], dtype=float)
    xs = np.asarray([float(p['x']) for p in pref], dtype=float)
    end = ts[-1]
    z = math.exp(-(end-ts[0])/tau) * 0.52 * xs[0]
    for k in range(1, len(pref)):
        dt = ts[k] - ts[k-1]
        z += 0.52 * 0.5 * (math.exp(-(end-ts[k-1])/tau)*xs[k-1] + math.exp(-(end-ts[k])/tau)*xs[k]) * dt / tau
    return z

def _predict_params(row, p):
    a, b, c, tau = map(float, p)
    x = float(row['x_now'])
    z = _z0(row, tau)
    u = float(row['u_now'])
    H = float(row['H'])
    n = max(1, int(math.ceil(H / 0.004)))
    h = H / n
    def f(xv, zv):
        return -a*xv + b*zv + c*u, (0.52*xv-zv)/tau
    for _ in range(n):
        k1x,k1z=f(x,z)
        k2x,k2z=f(x+0.5*h*k1x,z+0.5*h*k1z)
        k3x,k3z=f(x+0.5*h*k2x,z+0.5*h*k2z)
        k4x,k4z=f(x+h*k3x,z+h*k3z)
        x += h*(k1x+2*k2x+2*k3x+k4x)/6.0
        z += h*(k1z+2*k2z+2*k3z+k4z)/6.0
    return x

def fit(train_rows):
    global _params
    def residual(p):
        return np.asarray([_predict_params(r, p)-float(r['y']) for r in train_rows])
    best = None
    for start in ([0.5,0.5,0.5,0.2], [1.0,0.2,0.8,0.5], [0.2,1.0,0.2,1.0]):
        res = least_squares(residual, start, bounds=([1e-5,1e-5,1e-5,0.005],[10,10,10,10]), max_nfev=300)
        if best is None or np.dot(res.fun,res.fun) < np.dot(best.fun,best.fun):
            best = res
    _params = best.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('Call fit(train_rows) first')
    return float(_predict_params(row, _params))