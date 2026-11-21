import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _z0(row, tau):
    if row.get('mem_conv') is not None:
        return float(row['mem_conv'])
    p = row.get('raw_prefix') or []
    if not p:
        return 0.52 * float(row['x_now'])
    p = sorted(p, key=lambda q: q['t'])
    z = 0.52 * float(p[0]['x'])
    for a, b in zip(p[:-1], p[1:]):
        dt = float(b['t']) - float(a['t'])
        decay = math.exp(-dt / tau)
        xmid = 0.5 * (float(a['x']) + float(b['x']))
        z = decay * z + 0.52 * (1.0 - decay) * xmid
    return z

def _sim(row, pars):
    a, b, c, tau = [float(v) for v in pars]
    x = float(row['x_now'])
    z = _z0(row, tau)
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
    global _params
    ys = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_sim(r,p) for r in train_rows]) - ys
    res = least_squares(residual, x0=np.array([0.8, 0.5, 0.8, 0.15]),
                        bounds=([0.0, -5.0, -5.0, 0.005], [5.0, 5.0, 5.0, 5.0]),
                        max_nfev=3000)
    _params = res.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('Call fit(train_rows) first')
    return float(_sim(row, _params))