import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _history_z(row, tau):
    pref = row.get('raw_prefix')
    if not pref:
        if row.get('mem_conv') is not None:
            return float(row['mem_conv'])
        return 0.52 * float(row['x_now'])
    ts = np.array([float(p['t']) for p in pref], dtype=float)
    xs = np.array([float(p['x']) for p in pref], dtype=float)
    end = ts[-1]
    z = math.exp(-(end - ts[0]) / tau) * 0.52 * xs[0]
    conv = 0.0
    for k in range(1, len(ts)):
        dt = ts[k] - ts[k-1]
        conv += 0.5 * (math.exp(-(end-ts[k-1])/tau)*xs[k-1] + math.exp(-(end-ts[k])/tau)*xs[k]) * dt / tau
    return z + 0.52 * conv

def _forecast(row, pars):
    a, b, c, tau = [float(v) for v in pars]
    x = float(row['x_now'])
    z = _history_z(row, tau)
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
    ys = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_forecast(r, p) for r in train_rows]) - ys
    starts = ([0.3,0.2,0.3,0.15], [0.5,0.5,0.5,0.3], [1.0,0.8,0.8,0.8])
    best = None
    for start in starts:
        result = least_squares(residual, start, bounds=([1e-5,0.0,-5.0,0.01],[5.0,5.0,5.0,5.0]), max_nfev=3000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
        if best is None or np.dot(result.fun,result.fun) < np.dot(best.fun,best.fun):
            best = result
    _PARAMS = best.x
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return float(_forecast(row, _PARAMS))