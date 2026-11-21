import math
import numpy as np
from scipy.optimize import least_squares

def _z0(row, tau):
    pref = row.get('raw_prefix')
    if pref:
        z = 0.52 * float(pref[0]['x'])
        for p, q in zip(pref[:-1], pref[1:]):
            dt = float(q['t']) - float(p['t'])
            x0, x1 = float(p['x']), float(q['x'])
            # RK4 for tau*z' = .52*x(t)-z, with x linearly interpolated.
            def f(s, zz):
                x = x0 + (x1 - x0) * s
                return (0.52 * x - zz) / tau
            k1 = f(0.0, z)
            k2 = f(0.5, z + 0.5 * dt * k1)
            k3 = f(0.5, z + 0.5 * dt * k2)
            k4 = f(1.0, z + dt * k3)
            z += dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0
        return z
    if row.get('mem_conv') is not None:
        return float(row['mem_conv'])
    return 0.52 * float(row['x_now'])

def _sim(row, pars):
    a, b, c, tau = pars
    x = float(row['x_now'])
    z = _z0(row, tau)
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
        return np.array([_sim(r, p) - float(r['y']) for r in train_rows])
    best = None
    for start in ([0.4,0.2,0.8,0.15], [0.8,0.1,1.0,0.3], [0.2,0.5,0.5,0.08]):
        res = least_squares(residual, start, bounds=([0,0,-10,0.005],[10,10,10,5]), max_nfev=3000)
        if best is None or np.sum(res.fun**2) < np.sum(best.fun**2):
            best = res
    pars = best.x.tolist()
    def predict(row):
        return float(_sim(row, pars))
    return predict

def predict(row):
    raise RuntimeError('Call fit(train_rows) and use its returned prediction function.')