import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _z0(row):
    if row.get('mem_conv') is not None:
        return float(row['mem_conv'])
    hist = row.get('raw_prefix') or []
    if not hist:
        return 0.52 * float(row['x_now'])
    z = 0.52 * float(hist[0]['x'])
    t_prev = float(hist[0]['t'])
    x_prev = float(hist[0]['x'])
    # Integrate tau_fast dz/dt = 0.52*x(t)-z with linearly interpolated x.
    def deriv(zv, xv):
        return (0.52 * xv - zv)
    for item in hist[1:]:
        t = float(item['t']); x = float(item['x'])
        dt = t - t_prev
        if dt > 0:
            k1 = deriv(z, x_prev)
            k2 = deriv(z + 0.5*dt*k1, 0.5*(x_prev+x))
            k3 = deriv(z + 0.5*dt*k2, 0.5*(x_prev+x))
            k4 = deriv(z + dt*k3, x)
            z += dt*(k1 + 2*k2 + 2*k3 + k4)/6.0
        t_prev, x_prev = t, x
    return z

def _simulate(row, p):
    a, b, c, tau = [float(v) for v in p]
    x = float(row['x_now']); z = _z0(row); u = float(row['u_now'])
    H = float(row['H'])
    n = max(1, int(math.ceil(H / 0.004)))
    dt = H / n
    def f(xv, zv):
        return (-a*xv + b*zv + c*u, (0.52*xv - zv)/tau)
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
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([_simulate(r, p) for r in rows]) - y
    best = None
    for start in ([0.5,0.3,0.7,0.2], [1.0,0.5,1.0,0.1], [0.2,0.8,0.5,0.4]):
        res = least_squares(residual, start, bounds=([1e-5,0.0,0.0,0.005],[5.0,5.0,5.0,2.0]), max_nfev=3000)
        if best is None or np.dot(res.fun,res.fun) < np.dot(best.fun,best.fun):
            best = res
    _PARAMS = best.x
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_simulate(row, _PARAMS))