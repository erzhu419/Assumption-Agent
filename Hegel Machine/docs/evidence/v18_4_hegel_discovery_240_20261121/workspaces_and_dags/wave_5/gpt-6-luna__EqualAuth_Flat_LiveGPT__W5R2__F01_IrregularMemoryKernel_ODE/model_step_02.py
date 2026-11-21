import math
import numpy as np
from scipy.optimize import least_squares

def _z0(row, tau):
    if row.get('mem_conv') is not None:
        return float(row['mem_conv'])
    pref = row.get('raw_prefix') or []
    if not pref:
        return 0.52 * float(row['x_now'])
    z = 0.52 * float(pref[0]['x'])
    tprev = float(pref[0]['t'])
    xprev = float(pref[0]['x'])
    for p in pref[1:]:
        t = float(p['t']); x = float(p['x'])
        dt = max(0.0, t - tprev)
        n = max(1, int(math.ceil(dt / 0.004)))
        h = dt / n
        for i in range(n):
            xa = xprev + (x - xprev) * (i / n)
            xb = xprev + (x - xprev) * ((i + 1) / n)
            xm = 0.5 * (xa + xb)
            f = lambda zz, xx: (0.52 * xx - zz) / tau
            k1 = f(z, xa); k2 = f(z + 0.5*h*k1, xm)
            k3 = f(z + 0.5*h*k2, xm); k4 = f(z + h*k3, xb)
            z += h * (k1 + 2*k2 + 2*k3 + k4) / 6.0
        tprev, xprev = t, x
    return z

def _simulate(row, pars):
    a, b, c, tau = [float(v) for v in pars]
    x = float(row['x_now']); z = _z0(row, tau); u = float(row['u_now'])
    H = float(row['H'])
    n = max(1, int(math.ceil(H / 0.004))); h = H / n
    def f(xx, zz):
        return (-a*xx + b*zz + c*u, (0.52*xx - zz)/tau)
    for _ in range(n):
        k1x,k1z = f(x,z)
        k2x,k2z = f(x+0.5*h*k1x,z+0.5*h*k1z)
        k3x,k3z = f(x+0.5*h*k2x,z+0.5*h*k2z)
        k4x,k4z = f(x+h*k3x,z+h*k3z)
        x += h*(k1x+2*k2x+2*k3x+k4x)/6.0
        z += h*(k1z+2*k2z+2*k3z+k4z)/6.0
    return x

def fit(train_rows):
    def residual(p):
        return np.asarray([_simulate(r, p) - float(r['y']) for r in train_rows])
    result = least_squares(residual, x0=np.array([0.4, 0.3, 0.8, 1.0]),
                           bounds=([1e-5, 1e-5, 1e-5, 0.03], [5.0, 5.0, 5.0, 10.0]),
                           max_nfev=3000)
    return result.x

def predict(row):
    return float(_simulate(row, _MODEL_PARAMS))