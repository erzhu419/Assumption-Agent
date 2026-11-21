import math
import numpy as np
from scipy.optimize import least_squares

def _history_z(row, tau):
    pref = row.get('raw_prefix')
    if not pref:
        return float(row.get('mem_conv') or (0.52 * row['x_now']))
    times = np.array([p['t'] for p in pref], dtype=float)
    xs = np.array([p['x'] for p in pref], dtype=float)
    z = 0.52 * xs[0]
    for i in range(1, len(times)):
        dt = times[i] - times[i-1]
        x0, x1 = xs[i-1], xs[i]
        def f(s, x):
            return (0.52 * x - s) / tau
        k1 = f(z, x0)
        k2 = f(z + 0.5*dt*k1, 0.5*(x0+x1))
        k3 = f(z + 0.5*dt*k2, 0.5*(x0+x1))
        k4 = f(z + dt*k3, x1)
        z += dt*(k1 + 2*k2 + 2*k3 + k4)/6.0
    return z

def _forward(row, pars):
    a, b, c, tau = pars
    z = float(row['mem_conv']) if row.get('mem_conv') is not None else _history_z(row, tau)
    x = float(row['x_now'])
    u = float(row['u_now'])
    H = float(row['H'])
    n = max(1, int(math.ceil(H / 0.004)))
    dt = H / n
    def deriv(xx, zz):
        return -a*xx + b*zz + c*u, (0.52*xx - zz)/tau
    for _ in range(n):
        k1x,k1z = deriv(x,z)
        k2x,k2z = deriv(x+dt*k1x/2, z+dt*k1z/2)
        k3x,k3z = deriv(x+dt*k2x/2, z+dt*k2z/2)
        k4x,k4z = deriv(x+dt*k3x, z+dt*k3z)
        x += dt*(k1x+2*k2x+2*k3x+k4x)/6
        z += dt*(k1z+2*k2z+2*k3z+k4z)/6
    return x

def fit(train_rows):
    def residual(p):
        return np.array([_forward(r,p)-float(r['y']) for r in train_rows])
    starts = [[0.2,0.4,0.5,0.2],[0.5,0.8,0.8,0.5],[1.0,0.2,1.0,1.0]]
    best = None
    for s in starts:
        result = least_squares(residual, s, bounds=([0,0,-10,0.005],[10,10,10,10]), max_nfev=3000)
        if best is None or np.sum(result.fun**2) < np.sum(best.fun**2):
            best = result
    return best.x

def predict(row):
    return float(_forward(row, fit([row])))