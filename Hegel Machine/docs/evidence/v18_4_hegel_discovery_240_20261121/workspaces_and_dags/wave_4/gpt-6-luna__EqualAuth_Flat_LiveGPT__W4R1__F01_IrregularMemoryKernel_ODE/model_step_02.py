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
    ts = ts[-1] - ts
    z = 0.52 * xs[0]
    for i in range(1, len(ts)):
        dt = ts[i-1] - ts[i]
        e = math.exp(-dt / tau)
        z = e*z + 0.52 * (xs[i-1] + xs[i]) * 0.5 * (1.0-e)
    return z

def _pred(row, p):
    a, b, c, tau = map(float, p)
    x = float(row['x_now'])
    z = _z0(row, tau)
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
    def residual(p):
        return np.array([_pred(r,p)-float(r['y']) for r in train_rows])
    best = least_squares(residual, [0.3, 0.2, 0.5, 0.8], bounds=([0,0,0,0.03],[5,5,5,10]), max_nfev=3000)
    return best.x

def predict(row):
    return float(_pred(row, _MODEL_PARAMS))

_MODEL_PARAMS = None

def _fit_predictor(train_rows):
    global _MODEL_PARAMS
    _MODEL_PARAMS = fit(train_rows)
    return _MODEL_PARAMS

# Prediction uses the parameters returned by fit(train_rows); bind them before use.
