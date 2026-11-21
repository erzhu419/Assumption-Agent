import math
import numpy as np
from scipy.optimize import least_squares

_params = None
_kind = None

def _forecast(row, p, kind):
    a, b, v, rho, fifth = p[:5]
    if kind == 0:
        tau, c = fifth, 0.0
    elif kind == 1:
        tau, c = 1.0, fifth
    else:
        tau, c = fifth, p[5]
    u = float(row['u_now'])
    s = float(row['s_grad'])
    forcing = v * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + c * math.sqrt(max(u, 0.0))
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    h = float(row['H'])
    n = max(1, int(math.ceil(abs(h) / 0.01)))
    dt = h / n
    def deriv(x, z):
        return -a*x + b*z + forcing, (0.52*x-z)/tau
    for _ in range(n):
        k1, l1 = deriv(x, z)
        k2, l2 = deriv(x + dt*k1/2, z + dt*l1/2)
        k3, l3 = deriv(x + dt*k2/2, z + dt*l2/2)
        k4, l4 = deriv(x + dt*k3, z + dt*l3)
        x += dt*(k1 + 2*k2 + 2*k3 + k4)/6
        z += dt*(l1 + 2*l2 + 2*l3 + l4)/6
    return x

def _fit_kind(rows, kind):
    if kind == 0:
        start = [0.8, 0.4, 3.0, 0.7, 0.7]
        lo = [0.0, -4.0, 0.0, 0.0, 0.05]
        hi = [5.0, 4.0, 15.0, 1.5, 5.0]
    elif kind == 1:
        start = [0.8, 0.4, 3.0, 0.7, 0.2]
        lo = [0.0, -4.0, 0.0, 0.0, -5.0]
        hi = [5.0, 4.0, 15.0, 1.5, 5.0]
    else:
        start = [0.8, 0.4, 3.0, 0.7, 0.7, 0.2]
        lo = [0.0, -4.0, 0.0, 0.0, 0.05, -5.0]
        hi = [5.0, 4.0, 15.0, 1.5, 5.0, 5.0]
    def residual(p):
        return np.array([_forecast(r, p, kind) - float(r['y']) for r in rows])
    result = least_squares(residual, start, bounds=(lo, hi), max_nfev=500, ftol=1e-9, xtol=1e-9, gtol=1e-9)
    return result.x

def fit(train_rows):
    global _params, _kind
    rows = list(train_rows)
    candidates = []
    for kind in (0, 1, 2):
        p = _fit_kind(rows, kind)
        error = np.sqrt(np.mean([(_forecast(r, p, kind)-float(r['y']))**2 for r in rows]))
        candidates.append((error, kind, p))
    # Prefer a five-parameter mechanism unless an added branch materially improves fit.
    best5 = min(candidates[:2], key=lambda item: item[0])
    best = candidates[2] if candidates[2][0] < min(0.003, best5[0]*0.2) else best5
    _, _kind, _params = best

def predict(row):
    return float(_forecast(row, _params, _kind))