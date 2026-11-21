import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _z0(row, tau):
    if row.get('mem_conv') is not None:
        return float(row['mem_conv'])
    pts = row.get('raw_prefix') or []
    if not pts:
        return 0.52 * float(row['x_now'])
    z = 0.52 * float(pts[0]['x'])
    tprev = float(pts[0]['t'])
    xprev = float(pts[0]['x'])
    for p in pts[1:]:
        t = float(p['t']); x = float(p['x']); dt = t - tprev
        if dt > 0:
            e = math.exp(-dt / tau)
            # Exponential-kernel update with linearly interpolated x over this interval.
            z = z * e + 0.52 * (xprev * (1.0 - e) + (x - xprev) * (1.0 - tau * (1.0 - e) / dt))
        tprev, xprev = t, x
    return z

def _forward(row, pars):
    a, b, c, tau = pars
    x = float(row['x_now']); z = _z0(row, tau)
    u = float(row['u_now']); H = float(row['H'])
    n = max(1, int(math.ceil(H / 0.004)))
    dt = H / n
    def f(xv, zv):
        return -a*xv + b*zv + c*u, (0.52*xv-zv)/tau
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
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_forward(r,p) for r in train_rows]) - y
    starts = [np.array([0.5,0.5,0.5,0.3]), np.array([0.8,0.7,0.4,0.5]), np.array([0.3,0.9,0.6,0.8])]
    sols = [least_squares(residual,s,bounds=([0.001,0.001,0.001,0.02],[5,5,5,5]),max_nfev=3000) for s in starts]
    _PARAMS = min(sols,key=lambda q: np.sum(q.fun*q.fun)).x
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit must be called first')
    return float(_forward(row,_PARAMS))