import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _rk4_step(state, dt, deriv):
    k1 = deriv(state)
    k2 = deriv(state + 0.5 * dt * k1)
    k3 = deriv(state + 0.5 * dt * k2)
    k4 = deriv(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def _z_from_prefix(row, tau):
    if row.get('mem_conv') is not None:
        return float(row['mem_conv'])
    prefix = row.get('raw_prefix')
    if not prefix:
        return 0.52 * float(row['x_now'])
    ts = np.array([float(p['t']) for p in prefix], dtype=float)
    xs = np.array([float(p['x']) for p in prefix], dtype=float)
    z = 0.52 * xs[0]
    for i in range(len(ts)-1):
        dt = ts[i+1] - ts[i]
        n = max(1, int(math.ceil(dt / 0.004)))
        h = dt / n
        for j in range(n):
            x0 = xs[i] + (xs[i+1]-xs[i]) * (j/n)
            x1 = xs[i] + (xs[i+1]-xs[i]) * ((j+1)/n)
            xm = 0.5*(x0+x1)
            def dz(q, xval):
                return (0.52*xval-q)/tau
            k1 = dz(z, x0)
            k2 = dz(z + h*k1/2, xm)
            k3 = dz(z + h*k2/2, xm)
            k4 = dz(z + h*k3, x1)
            z += h*(k1+2*k2+2*k3+k4)/6
    return float(z)

def _forward(row, p):
    a, b, c, tau = p
    z0 = _z_from_prefix(row, tau)
    state = np.array([float(row['x_now']), z0], dtype=float)
    H = float(row['H'])
    n = max(1, int(math.ceil(H / 0.004)))
    dt = H/n
    u = float(row['u_now'])
    def deriv(q):
        x, z = q
        return np.array([-a*x + b*z + c*u, (0.52*x-z)/tau])
    for _ in range(n):
        state = _rk4_step(state, dt, deriv)
    return float(state[0])

def fit(train_rows):
    global _PARAMS
    def residual(p):
        return np.array([_forward(r, p)-float(r['y']) for r in train_rows])
    starts = [np.array([0.5,0.2,0.5,1.0]), np.array([0.8,0.1,0.8,0.5])]
    best = None
    for x0 in starts:
        res = least_squares(residual, x0, bounds=([1e-5,1e-5,1e-5,0.03],[5,5,5,10]), max_nfev=3000)
        if best is None or np.dot(res.fun,res.fun) < np.dot(best.fun,best.fun):
            best = res
    _PARAMS = best.x.copy()
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return _forward(row, _PARAMS)
