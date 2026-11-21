import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _z_history(row, tau):
    if row.get('mem_conv') is not None:
        return float(row['mem_conv'])
    pref = row.get('raw_prefix') or []
    if not pref:
        return 0.52 * float(row['x_now'])
    times = np.array([float(p['t']) for p in pref], dtype=float)
    xs = np.array([float(p['x']) for p in pref], dtype=float)
    end = times[-1]
    z = math.exp(-(end-times[0])/tau) * 0.52 * xs[0]
    for k in range(1, len(times)):
        dt = times[k] - times[k-1]
        z += 0.52 * 0.5 * (math.exp(-(end-times[k-1])/tau)*xs[k-1] + math.exp(-(end-times[k])/tau)*xs[k]) * dt / tau
    return z

def _pred(row, p):
    a, b, c, tau = p
    x = float(row['x_now'])
    z = _z_history(row, tau)
    u = float(row['u_now'])
    H = float(row['H'])
    n = max(1, int(math.ceil(H / 0.004)))
    h = H / n
    def f(x, z):
        return -a*x + b*z + c*u, (0.52*x-z)/tau
    for _ in range(n):
        k1x,k1z=f(x,z)
        k2x,k2z=f(x+h*k1x/2,z+h*k1z/2)
        k3x,k3z=f(x+h*k2x/2,z+h*k2z/2)
        k4x,k4z=f(x+h*k3x,z+h*k3z)
        x += h*(k1x+2*k2x+2*k3x+k4x)/6
        z += h*(k1z+2*k2z+2*k3z+k4z)/6
    return x

def fit(train_rows):
    global _params
    ys = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_pred(r, p) for r in train_rows]) - ys
    best = None
    for start in ([0.5,0.4,0.5,0.2], [1.0,0.8,0.8,0.5], [0.2,0.2,0.3,1.0]):
        result = least_squares(residual, start, bounds=([0,0,0,0.01],[5,5,5,5]), max_nfev=3000)
        if best is None or np.dot(result.fun,result.fun) < np.dot(best.fun,best.fun):
            best = result
    _params = best.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return float(_pred(row, _params))