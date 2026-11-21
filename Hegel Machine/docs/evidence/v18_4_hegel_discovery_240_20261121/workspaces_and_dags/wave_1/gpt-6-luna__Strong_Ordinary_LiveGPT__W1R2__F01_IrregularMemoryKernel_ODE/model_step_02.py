import math
import numpy as np
from scipy.optimize import least_squares

def _z0(row, tau):
    prefix = row.get('raw_prefix') or []
    if not prefix:
        return float(row.get('mem_conv') if row.get('mem_conv') is not None else 0.52 * row['x_now'])
    tlast = float(prefix[-1]['t'])
    z = 0.52 * float(prefix[0]['x'])
    for a, b in zip(prefix[:-1], prefix[1:]):
        dt = max(0.0, tlast - float(a['t'])) - max(0.0, tlast - float(b['t']))
        decay = math.exp(-dt / tau)
        xa, xb = float(a['x']), float(b['x'])
        z = decay * z + 0.52 * (1.0 - decay) * 0.5 * (xa + xb)
    return z

def _sim(row, p):
    a, b, c, tau = p
    x = float(row['x_now'])
    z = _z0(row, tau)
    u = float(row['u_now'])
    H = float(row['H'])
    n = max(1, int(math.ceil(H / 0.004)))
    dt = H / n
    def f(x, z):
        return -a*x + b*z + c*u, (0.52*x-z)/tau
    for _ in range(n):
        k1x,k1z=f(x,z)
        k2x,k2z=f(x+dt*k1x/2,z+dt*k1z/2)
        k3x,k3z=f(x+dt*k2x/2,z+dt*k2z/2)
        k4x,k4z=f(x+dt*k3x,z+dt*k3z)
        x += dt*(k1x+2*k2x+2*k3x+k4x)/6
        z += dt*(k1z+2*k2z+2*k3z+k4z)/6
    return x

def fit(train_rows):
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([_sim(r, p) for r in rows]) - y
    starts = ([0.5,0.5,0.5,0.1], [1.0,1.0,0.5,0.3], [0.2,0.2,1.0,0.05])
    fits = [least_squares(residual, s, bounds=([0,0,0,0.005],[10,10,10,5]), max_nfev=3000) for s in starts]
    return min(fits, key=lambda q: np.dot(q.fun,q.fun)).x

def predict(row):
    return float(_sim(row, fit._params))
