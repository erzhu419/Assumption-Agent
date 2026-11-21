import math
import numpy as np
from scipy.optimize import least_squares

P = None

def _z0(row, tau):
    if row.get('mem_conv') is not None:
        return float(row['mem_conv'])
    pref = row.get('raw_prefix') or []
    if len(pref) < 2:
        return 0.52 * float(row['x_now'])
    t0 = float(pref[0]['t'])
    te = float(pref[-1]['t'])
    z = 0.52 * float(pref[0]['x']) * math.exp(-(te-t0)/tau)
    conv = 0.0
    for left, right in zip(pref[:-1], pref[1:]):
        tl, tr = float(left['t']), float(right['t'])
        dt = tr - tl
        conv += 0.5 * (math.exp(-(te-tl)/tau) * float(left['x']) + math.exp(-(te-tr)/tau) * float(right['x'])) * dt / tau
    return z + 0.52 * conv

def _integrate(row, pars):
    a, b, c, tau = map(float, pars)
    x = float(row['x_now'])
    z = _z0(row, tau)
    H = float(row['H'])
    u = float(row['u_now'])
    n = max(1, int(math.ceil(H / 0.004)))
    h = H / n
    def f(xx, zz):
        return -a*xx + b*zz + c*u, (0.52*xx-zz)/tau
    for _ in range(n):
        k1x,k1z=f(x,z)
        k2x,k2z=f(x+0.5*h*k1x,z+0.5*h*k1z)
        k3x,k3z=f(x+0.5*h*k2x,z+0.5*h*k2z)
        k4x,k4z=f(x+h*k3x,z+h*k3z)
        x += h*(k1x+2*k2x+2*k3x+k4x)/6.0
        z += h*(k1z+2*k2z+2*k3z+k4z)/6.0
    return x

def fit(train_rows):
    global P
    def residual(q):
        return np.asarray([_integrate(r, q) - float(r['y']) for r in train_rows], dtype=float)
    starts = [np.array([0.4, 0.3, 0.5, 0.2]), np.array([0.8, 0.5, 0.8, 0.5]), np.array([0.2, 0.1, 0.3, 1.0])]
    lo = np.array([1e-5, 1e-5, 1e-5, 0.01])
    hi = np.array([5.0, 5.0, 5.0, 5.0])
    results = [least_squares(residual, s, bounds=(lo,hi), max_nfev=3000) for s in starts]
    P = min(results, key=lambda r: float(np.dot(r.fun,r.fun))).x
    return P

def predict(row):
    if P is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return float(_integrate(row, P))