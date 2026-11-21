import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _simulate(rows, p):
    a, b, c, tau = np.exp(np.asarray(p, dtype=float))
    n = len(rows)
    x = np.array([float(r['x_now']) for r in rows])
    z = np.array([float(r['mem_conv']) if r.get('mem_conv') is not None else 0.52*float(r['x_now']) for r in rows])
    u = np.array([float(r['u_now']) for r in rows])
    H = np.array([float(r['H']) for r in rows])
    steps = np.maximum(1, np.ceil(H / 0.05).astype(int))
    max_steps = int(steps.max())
    for j in range(max_steps):
        active = j < steps
        dt = np.where(active, H / steps, 0.0)
        def deriv(xx, zz):
            return -a*xx + b*zz + c*u, (0.52*xx - zz)/tau
        k1x,k1z = deriv(x,z)
        k2x,k2z = deriv(x+0.5*dt*k1x,z+0.5*dt*k1z)
        k3x,k3z = deriv(x+0.5*dt*k2x,z+0.5*dt*k2z)
        k4x,k4z = deriv(x+dt*k3x,z+dt*k3z)
        x = x + dt*(k1x+2*k2x+2*k3x+k4x)/6.0
        z = z + dt*(k1z+2*k2z+2*k3z+k4z)/6.0
    return x

def fit(train_rows):
    global _PARAMS
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return _simulate(train_rows, p) - y
    result = least_squares(residual, np.log([0.08, 0.1, 0.5, 0.15]), max_nfev=180, bounds=(np.log([1e-4,1e-5,1e-5,0.01]), np.log([3.0,3.0,20.0,5.0])))
    _PARAMS = result.x.copy()
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return float(_simulate([row], _PARAMS)[0])