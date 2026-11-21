import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _integrate(row, p):
    a_slow, b_couple, vmax, rho, tau_fast = p
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    s = float(row['s_grad'])
    u_n = float(row['u_now']) / 2.0
    H = float(row['H'])
    j_siph = vmax * (1.0 - rho * s / (1.0 + s)) * (u_n / (1.2 + u_n))
    n = max(1, int(math.ceil(H * 200)))
    dt = H / n
    def deriv(xv, zv):
        return -a_slow * xv + b_couple * zv + j_siph, (0.52 * xv - zv) / tau_fast
    for _ in range(n):
        k1x, k1z = deriv(x, z)
        k2x, k2z = deriv(x + 0.5*dt*k1x, z + 0.5*dt*k1z)
        k3x, k3z = deriv(x + 0.5*dt*k2x, z + 0.5*dt*k2z)
        k4x, k4z = deriv(x + dt*k3x, z + dt*k3z)
        x += dt*(k1x + 2*k2x + 2*k3x + k4x)/6.0
        z += dt*(k1z + 2*k2z + 2*k3z + k4z)/6.0
    return x

def fit(train_rows):
    global _PARAMS
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_integrate(r, p) for r in train_rows]) - y
    lower = np.array([0.001, 0.0, 0.01, 0.0, 0.02])
    upper = np.array([5.0, 5.0, 20.0, 0.999, 10.0])
    starts = [np.array([a, b, 6.5, 0.70, t]) for a,b,t in [(0.5,0.4,0.5),(1.0,0.8,1.0),(0.2,0.2,0.2),(2.0,1.5,2.0)]]
    results = [least_squares(residual, np.clip(q, lower+1e-8, upper-1e-8), bounds=(lower, upper), max_nfev=3000) for q in starts]
    best = min(results, key=lambda z: float(np.dot(z.fun, z.fun)))
    _PARAMS = best.x
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return float(_integrate(row, _PARAMS))