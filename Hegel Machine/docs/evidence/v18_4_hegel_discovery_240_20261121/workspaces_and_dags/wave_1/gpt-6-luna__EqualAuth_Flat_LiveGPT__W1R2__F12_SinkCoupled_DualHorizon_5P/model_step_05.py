import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _simulate(row, p):
    a_slow, b_couple, c_drive, tau_fast = p
    H = max(0.0, float(row['H']))
    n = max(1, int(math.ceil(H / 0.004)))
    dt = H / n if n else 0.0
    x = float(row['x_now'])
    z = float(row.get('mem_conv') or 0.0)
    u = float(row['u_now'])
    def deriv(xv, zv):
        return (-a_slow*xv + b_couple*zv + c_drive*u,
                (0.52*xv - zv) / tau_fast)
    for _ in range(n):
        k1x,k1z = deriv(x,z)
        k2x,k2z = deriv(x+0.5*dt*k1x,z+0.5*dt*k1z)
        k3x,k3z = deriv(x+0.5*dt*k2x,z+0.5*dt*k2z)
        k4x,k4z = deriv(x+dt*k3x,z+dt*k3z)
        x += dt*(k1x+2*k2x+2*k3x+k4x)/6.0
        z += dt*(k1z+2*k2z+2*k3z+k4z)/6.0
    return x

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows])
    def unpack(q):
        return np.array([math.exp(q[0]), q[1], q[2], math.exp(q[3])])
    def residual(q):
        p = unpack(q)
        return np.asarray([_simulate(r,p) for r in rows]) - y
    starts = [np.array([math.log(.1), .1, .5, math.log(1.)]),
              np.array([math.log(.5), -.1, 1., math.log(3.)])]
    sols = [least_squares(residual, q, max_nfev=500) for q in starts]
    best = min(sols, key=lambda s: float(np.dot(s.fun,s.fun)))
    _PARAMS = unpack(best.x)
    return _PARAMS.tolist()

def predict(row):
    if _PARAMS is None:
        raise ValueError('fit(train_rows) must be called first')
    return float(_simulate(row, _PARAMS))