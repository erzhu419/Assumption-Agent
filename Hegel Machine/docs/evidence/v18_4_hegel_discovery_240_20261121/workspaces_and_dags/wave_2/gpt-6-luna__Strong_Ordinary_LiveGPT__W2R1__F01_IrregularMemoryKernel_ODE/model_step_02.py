import math
import numpy as np
from scipy.optimize import least_squares

def _z0(row):
    pref = row.get('raw_prefix') or []
    if not pref:
        return 0.52 * float(row['x_now'])
    z = 0.52 * float(pref[0]['x'])
    tprev = float(pref[0]['t'])
    xprev = float(pref[0]['x'])
    for p in pref[1:]:
        t = float(p['t']); x = float(p['x'])
        dt = t - tprev
        e = math.exp(-dt / TAU_FAST)
        z = z * e + 0.52 * (xprev * (1.0 - e) + (x - xprev) * (1.0 - TAU_FAST * (1.0 - e) / dt))
        tprev, xprev = t, x
    return z

TAU_FAST = 0.08

def _integrate(row, pars):
    a, b, c, tau = pars
    x = float(row['x_now'])
    zinit = row.get('mem_conv')
    z = float(zinit) if zinit is not None else _z0(row)
    u = float(row['u_now'])
    H = float(row['H'])
    n = max(1, int(math.ceil(H / 0.004)))
    dt = H / n
    def f(xv, zv):
        return (-a*xv + b*zv + c*u, (0.52*xv-zv)/tau)
    for _ in range(n):
        k1x,k1z=f(x,z)
        k2x,k2z=f(x+0.5*dt*k1x,z+0.5*dt*k1z)
        k3x,k3z=f(x+0.5*dt*k2x,z+0.5*dt*k2z)
        k4x,k4z=f(x+dt*k3x,z+dt*k3z)
        x += dt*(k1x+2*k2x+2*k3x+k4x)/6.0
        z += dt*(k1z+2*k2z+2*k3z+k4z)/6.0
    return x

def fit(train_rows):
    global TAU_FAST
    def residual(p):
        return np.array([_integrate(r,p) - float(r['y']) for r in train_rows])
    best = None
    for start in ([0.5,0.4,0.5,0.15],[1.0,0.2,1.0,0.3],[0.2,0.8,0.2,0.06]):
        res = least_squares(residual, start, bounds=([0.001,-3.0,-3.0,0.005],[5.0,3.0,3.0,2.0]), max_nfev=3000)
        if best is None or np.sum(res.fun**2) < np.sum(best.fun**2): best = res
    TAU_FAST = float(best.x[3])
    return [float(v) for v in best.x]

def predict(row):
    raise RuntimeError('predict requires fitted parameters; use fit_params via the returned fit vector')