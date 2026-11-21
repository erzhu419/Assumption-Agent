import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = np.array([0.45, 0.35, 0.55, 0.25], dtype=float)

def _z0(row, tau):
    if row.get('mem_conv') is not None:
        return float(row['mem_conv'])
    pref = row.get('raw_prefix') or []
    if not pref:
        return 0.52 * float(row['x_now'])
    end = float(pref[-1]['t'])
    span = end - float(pref[0]['t'])
    z = 0.52 * float(pref[0]['x']) * math.exp(-span / tau)
    for a, b in zip(pref[:-1], pref[1:]):
        dt = float(b['t']) - float(a['t'])
        xa, xb = float(a['x']), float(b['x'])
        ta, tb = end - float(a['t']), end - float(b['t'])
        z += 0.52 * 0.5 * (math.exp(-ta / tau) * xa + math.exp(-tb / tau) * xb) * dt / tau
    return z

def _forward(row, p):
    a, b, c, tau = [float(v) for v in p]
    tau = max(tau, 1e-6)
    x, z = float(row['x_now']), _z0(row, tau)
    u, H = float(row['u_now']), float(row['H'])
    n = max(1, int(math.ceil(H / 0.004)))
    dt = H / n
    def f(q):
        xx, zz = q
        return np.array([-a*xx + b*zz + c*u, (0.52*xx-zz)/tau])
    q = np.array([x, z], dtype=float)
    for _ in range(n):
        k1=f(q); k2=f(q+0.5*dt*k1); k3=f(q+0.5*dt*k2); k4=f(q+dt*k3)
        q += dt*(k1+2*k2+2*k3+k4)/6.0
    return float(q[0])

def fit(train_rows):
    global _PARAMS
    rows = [r for r in train_rows if r.get('y') is not None]
    if rows:
        def residual(p):
            return np.array([_forward(r,p)-float(r['y']) for r in rows])
        result = least_squares(residual, _PARAMS, bounds=([0.001,-3.0,-3.0,0.01],[5.0,3.0,3.0,5.0]), max_nfev=300)
        _PARAMS = result.x
    return _PARAMS.copy()

def predict(row):
    return _forward(row, _PARAMS)
