import math
import numpy as np
from scipy.optimize import least_squares

def _z0(row, tau):
    if row.get('mem_conv') is not None:
        return float(row['mem_conv'])
    pref = row.get('raw_prefix') or []
    if not pref:
        return 0.52 * float(row['x_now'])
    ts = np.array([float(p['t']) for p in pref], dtype=float)
    xs = np.array([float(p['x']) for p in pref], dtype=float)
    z = 0.52 * xs[0]
    # Integrate the prescribed exponentially relaxing memory state over the
    # observed, irregularly sampled x trajectory using linear interpolation.
    for j in range(len(ts)-1):
        dt = ts[j+1] - ts[j]
        n = max(1, int(math.ceil(dt / 0.004)))
        h = dt / n
        for k in range(n):
            f0 = k / n
            f1 = (k + 0.5) / n
            f2 = (k + 1) / n
            x0 = xs[j] + f0 * (xs[j+1] - xs[j])
            xm = xs[j] + f1 * (xs[j+1] - xs[j])
            x1 = xs[j] + f2 * (xs[j+1] - xs[j])
            def dz(x, zz):
                return (0.52*x - zz) / tau
            k1 = dz(x0, z)
            k2 = dz(xm, z + 0.5*h*k1)
            k3 = dz(xm, z + 0.5*h*k2)
            k4 = dz(x1, z + h*k3)
            z += h*(k1 + 2*k2 + 2*k3 + k4)/6
    return z

def _forward(row, p):
    a, b, c, tau = p
    x = float(row['x_now'])
    z = _z0(row, tau)
    u = float(row['u_now'])
    H = float(row['H'])
    n = max(1, int(math.ceil(H / 0.004)))
    h = H / n
    def rhs(xx, zz):
        return -a*xx + b*zz + c*u, (0.52*xx - zz)/tau
    for _ in range(n):
        k1x,k1z = rhs(x,z)
        k2x,k2z = rhs(x+0.5*h*k1x,z+0.5*h*k1z)
        k3x,k3z = rhs(x+0.5*h*k2x,z+0.5*h*k2z)
        k4x,k4z = rhs(x+h*k3x,z+h*k3z)
        x += h*(k1x+2*k2x+2*k3x+k4x)/6
        z += h*(k1z+2*k2z+2*k3z+k4z)/6
    return x

def fit(train_rows):
    def residual(p):
        return np.array([_forward(r,p)-float(r['y']) for r in train_rows])
    starts = [[0.5,0.5,0.5,0.5],[1.0,0.3,0.5,1.0],[0.2,0.8,0.8,2.0]]
    fits = [least_squares(residual, s, bounds=([0,0,0,0.03],[5,5,5,10]), max_nfev=3000) for s in starts]
    return min(fits, key=lambda q: float(np.dot(q.fun,q.fun))).x.tolist()

def predict(row):
    return float(_forward(row, _PARAMS))