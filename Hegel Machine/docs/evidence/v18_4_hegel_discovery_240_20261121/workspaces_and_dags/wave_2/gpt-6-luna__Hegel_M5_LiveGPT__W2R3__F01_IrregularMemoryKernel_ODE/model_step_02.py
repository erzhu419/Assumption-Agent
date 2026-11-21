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
    ts = np.array([float(p['t']) for p in pref])
    xs = np.array([float(p['x']) for p in pref])
    end = ts[-1]
    z = math.exp(-(end-ts[0])/tau) * 0.52 * xs[0]
    conv = 0.0
    for k in range(1, len(ts)):
        d = ts[k] - ts[k-1]
        conv += 0.5 * (math.exp(-(end-ts[k-1])/tau)*xs[k-1] + math.exp(-(end-ts[k])/tau)*xs[k]) * d / tau
    return z + 0.52 * conv

def _predict_params(row, p):
    a, b, c, tau = map(float, p)
    x = float(row['x_now'])
    z = _history_z(row, tau)
    u = float(row['u_now'])
    H = float(row['H'])
    n = max(1, int(math.ceil(H / 0.004)))
    dt = H / n
    def f(xx, zz):
        return -a*xx + b*zz + c*u, (0.52*xx - zz)/tau
    for _ in range(n):
        k1x,k1z=f(x,z)
        k2x,k2z=f(x+dt*k1x/2,z+dt*k1z/2)
        k3x,k3z=f(x+dt*k2x/2,z+dt*k2z/2)
        k4x,k4z=f(x+dt*k3x,z+dt*k3z)
        x += dt*(k1x+2*k2x+2*k3x+k4x)/6
        z += dt*(k1z+2*k2z+2*k3z+k4z)/6
    return x

def fit(train_rows):
    global _PARAMS
    def residual(p):
        return np.array([_predict_params(r, p) - float(r['y']) for r in train_rows])
    starts = [np.array([0.5, 0.5, 0.5, 0.2]), np.array([1.0, 0.2, 1.0, 0.5]), np.array([0.2, 1.0, 0.2, 1.0])]
    best = None
    for x0 in starts:
        result = least_squares(residual, x0, bounds=([0.001, 0.001, 0.001, 0.01], [5, 5, 5, 5]), max_nfev=3000)
        if best is None or np.sum(result.fun**2) < np.sum(best.fun**2):
            best = result
    _PARAMS = best.x
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return float(_predict_params(row, _PARAMS))