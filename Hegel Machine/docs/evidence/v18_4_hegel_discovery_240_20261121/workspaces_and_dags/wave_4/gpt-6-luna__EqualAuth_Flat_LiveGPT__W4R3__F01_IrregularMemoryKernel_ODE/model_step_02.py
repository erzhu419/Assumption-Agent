import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _z0(row, tau):
    pref = row.get('raw_prefix')
    if row.get('mem_conv') is not None:
        return float(row['mem_conv'])
    if not pref:
        return 0.52 * float(row['x_now'])
    ts = np.array([p['t'] for p in pref], dtype=float)
    xs = np.array([p['x'] for p in pref], dtype=float)
    z = 0.52 * xs[0]
    for i in range(1, len(ts)):
        dt = ts[i] - ts[i-1]
        if dt <= 0: continue
        # RK4 for dz/dt=(0.52*x(t)-z)/tau with linear x interpolation.
        n = max(1, int(math.ceil(dt / 0.004)))
        h = dt / n
        x0, x1 = xs[i-1], xs[i]
        for j in range(n):
            xa = x0 + (x1-x0) * (j/n)
            xb = x0 + (x1-x0) * ((j+1)/n)
            xm = (xa+xb)/2
            f = lambda zz, xx: (0.52*xx-zz)/tau
            k1 = f(z, xa); k2 = f(z+h*k1/2, xm)
            k3 = f(z+h*k2/2, xm); k4 = f(z+h*k3, xb)
            z += h*(k1+2*k2+2*k3+k4)/6
    return z

def _forecast(row, p):
    a, b, c, tau = p
    x = float(row['x_now'])
    z = _z0(row, tau)
    H = float(row['H'])
    n = max(1, int(math.ceil(H/0.004)))
    h = H/n
    def deriv(xv, zv):
        return -a*xv+b*zv+c*float(row['u_now']), (0.52*xv-zv)/tau
    for _ in range(n):
        k1x,k1z=deriv(x,z)
        k2x,k2z=deriv(x+h*k1x/2,z+h*k1z/2)
        k3x,k3z=deriv(x+h*k2x/2,z+h*k2z/2)
        k4x,k4z=deriv(x+h*k3x,z+h*k3z)
        x += h*(k1x+2*k2x+2*k3x+k4x)/6
        z += h*(k1z+2*k2z+2*k3z+k4z)/6
    return x

def fit(train_rows):
    global _PARAMS
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        return np.array([_forecast(r,p) for r in train_rows])-y
    result = least_squares(residual, x0=[0.5,0.5,0.5,1.0],
                           bounds=([1e-5,1e-5,1e-5,0.02],[5.0,5.0,5.0,10.0]),
                           max_nfev=3000)
    _PARAMS = result.x
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return float(_forecast(row, _PARAMS))