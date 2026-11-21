import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _initial_z(row, tau):
    if row.get('mem_conv') is not None:
        return float(row['mem_conv'])
    pref = row.get('raw_prefix') or []
    if len(pref) < 2:
        return 0.52 * float(row['x_now'])
    ts = np.array([float(p['t']) for p in pref], dtype=float)
    xs = np.array([float(p['x']) for p in pref], dtype=float)
    end = ts[-1]
    conv = 0.0
    for k in range(1, len(ts)):
        dt = ts[k] - ts[k-1]
        conv += 0.5 * (math.exp(-(end-ts[k-1])/tau)*xs[k-1] + math.exp(-(end-ts[k])/tau)*xs[k]) * dt / tau
    return math.exp(-(end-ts[0])/tau) * 0.52 * xs[0] + 0.52 * conv

def _forecast(row, pars):
    a, b, c, tau = [float(v) for v in pars]
    x = float(row['x_now'])
    z = _initial_z(row, tau)
    u = float(row['u_now'])
    H = float(row['H'])
    n = max(1, int(math.ceil(H / 0.004)))
    dt = H / n
    def deriv(xx, zz):
        return (-a*xx + b*zz + c*u, (0.52*xx - zz)/tau)
    for _ in range(n):
        k1x, k1z = deriv(x, z)
        k2x, k2z = deriv(x+0.5*dt*k1x, z+0.5*dt*k1z)
        k3x, k3z = deriv(x+0.5*dt*k2x, z+0.5*dt*k2z)
        k4x, k4z = deriv(x+dt*k3x, z+dt*k3z)
        x += dt*(k1x+2*k2x+2*k3x+k4x)/6.0
        z += dt*(k1z+2*k2z+2*k3z+k4z)/6.0
    return x

def fit(train_rows):
    global _PARAMS
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_forecast(r, p) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([0.8, 0.35, 0.9, 0.25]), bounds=([0.001, 0.001, 0.001, 0.005], [5.0, 5.0, 5.0, 5.0]), max_nfev=3000)
    _PARAMS = result.x
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    return float(_forecast(row, _PARAMS))