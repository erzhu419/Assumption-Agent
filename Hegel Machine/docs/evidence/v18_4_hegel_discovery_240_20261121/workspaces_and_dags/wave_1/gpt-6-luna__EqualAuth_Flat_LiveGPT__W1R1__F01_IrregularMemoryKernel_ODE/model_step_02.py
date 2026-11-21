import math
import numpy as np
from scipy.optimize import least_squares

def _z0(row):
    pref = row.get('raw_prefix') or []
    if not pref:
        return 0.52 * float(row.get('x_now', 0.0))
    ts = np.array([float(p['t']) for p in pref], dtype=float)
    xs = np.array([float(p['x']) for p in pref], dtype=float)
    us = np.array([float(p['u']) for p in pref], dtype=float)
    order = np.argsort(ts)
    ts, xs, us = ts[order], xs[order], us[order]
    # Exponential kernel integration with z initialized at 0.52*x(t0).
    tau = _FIT_TAU[0] if _FIT_TAU else 1.0
    z = 0.52 * xs[0]
    for i in range(1, len(ts)):
        dt = ts[i] - ts[i-1]
        # trapezoidal forcing over each interval
        f0, f1 = 0.52 * xs[i-1], 0.52 * xs[i]
        e = math.exp(-dt / tau)
        z = e*z + 0.5*(f0+f1)*(1.0-e)
    return z

_FIT_TAU = []

def _predict_params(row, p):
    a, b, c, tau = [float(v) for v in p]
    pref = row.get('raw_prefix') or []
    if row.get('mem_conv') is not None:
        z = float(row['mem_conv'])
    elif pref:
        ts = np.array([float(q['t']) for q in pref], dtype=float)
        xs = np.array([float(q['x']) for q in pref], dtype=float)
        order = np.argsort(ts); ts, xs = ts[order], xs[order]
        z = 0.52 * xs[0]
        for i in range(1, len(ts)):
            dt = ts[i]-ts[i-1]
            e = math.exp(-dt/tau)
            z = e*z + 0.26*(xs[i-1]+xs[i])*(1.0-e)
    else:
        z = 0.52 * float(row['x_now'])
    x = float(row['x_now']); u = float(row['u_now']); H = float(row['H'])
    n = max(1, int(math.ceil(H/0.004))); dt = H/n
    def deriv(xv, zv):
        return -a*xv+b*zv+c*u, (0.52*xv-zv)/tau
    for _ in range(n):
        k1x,k1z=deriv(x,z)
        k2x,k2z=deriv(x+dt*k1x/2,z+dt*k1z/2)
        k3x,k3z=deriv(x+dt*k2x/2,z+dt*k2z/2)
        k4x,k4z=deriv(x+dt*k3x,z+dt*k3z)
        x += dt*(k1x+2*k2x+2*k3x+k4x)/6
        z += dt*(k1z+2*k2z+2*k3z+k4z)/6
    return x

def fit(train_rows):
    rows = list(train_rows)
    def residual(p):
        return np.array([_predict_params(r,p)-float(r['y']) for r in rows])
    starts = [np.array([0.3,0.5,0.5,0.2]), np.array([0.5,0.5,0.5,0.5]), np.array([0.2,0.8,0.3,1.0])]
    best = None
    for x0 in starts:
        res = least_squares(residual, x0, bounds=([0,0,0,0.01],[5,5,5,10]), max_nfev=3000)
        if best is None or np.dot(res.fun,res.fun)<np.dot(best.fun,best.fun): best=res
    return best.x

def predict(row):
    return float(_predict_params(row, _MODEL_PARAMS))

_MODEL_PARAMS = np.array([0.3,0.5,0.5,0.2])

def fit(train_rows):
    global _MODEL_PARAMS
    rows = list(train_rows)
    def residual(p):
        return np.array([_predict_params(r,p)-float(r['y']) for r in rows])
    starts=[np.array([0.3,0.5,0.5,0.2]),np.array([0.5,0.5,0.5,0.5]),np.array([0.2,0.8,0.3,1.0])]
    best=None
    for x0 in starts:
        res=least_squares(residual,x0,bounds=([0,0,0,0.01],[5,5,5,10]),max_nfev=3000)
        if best is None or np.dot(res.fun,res.fun)<np.dot(best.fun,best.fun): best=res
    _MODEL_PARAMS=best.x
    return _MODEL_PARAMS
