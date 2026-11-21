import math
import numpy as np
from scipy.optimize import least_squares

def _value(row, p):
    vmax, rho, c_loc, k_aval, s_crit = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    return base + c_loc * math.sqrt(max(u, 0.0)) - k_aval * max(0.0, s - s_crit) * u

def fit(train_rows):
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows], dtype=float)
    lo = np.array([0.01, 0.0, -3.0, 0.0, 2.0])
    hi = np.array([30.0, 0.999, 3.0, 10.0, 3.2])
    best = None
    for start in ([6.5, 0.70, 0.2, 1.0, 2.58], [8.0, 0.72, -0.2, 1.5, 2.5], [5.0, 0.65, 0.5, 0.7, 2.8]):
        result = least_squares(lambda p: np.asarray([_value(r, p) for r in rows]) - y, np.asarray(start, dtype=float), bounds=(lo, hi), max_nfev=10000)
        if best is None or np.sum(result.fun ** 2) < np.sum(best.fun ** 2):
            best = result
    p = best.x
    return {'vmax': float(p[0]), 'rho': float(p[1]), 'c_loc': float(p[2]), 'k_aval': float(p[3]), 's_crit': float(p[4])}

def predict(row):
    p = [float(row.get(k, d)) for k, d in [('vmax', 6.5), ('rho', 0.70), ('c_loc', 0.2), ('k_aval', 1.0), ('s_crit', 2.58)]]
    return float(_value(row, p))