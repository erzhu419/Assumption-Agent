import math
import numpy as np
from scipy.optimize import least_squares

def _features(row):
    u = max(1e-8, float(row['u_now']))
    s = max(0.0, float(row['s_grad']))
    return u, s

def _predict_params(p, row):
    u, s = _features(row)
    base, amp, vmax, km, rho, kval, scrit = p
    saturation = u / (km + u)
    sink = 1.0 - rho * s / (1.0 + s)
    avalanche = kval * max(0.0, s - scrit) * u
    return base + amp * math.sqrt(u) + vmax * saturation * sink - avalanche

def fit(train_rows):
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    starts = [
        [0.0, 18.0, 8.0, 0.02, 0.5, 2.0, 2.5],
        [0.0, 25.0, 5.0, 0.01, 0.7, 1.0, 2.6],
        [0.0, 12.0, 10.0, 0.04, 0.3, 4.0, 2.4],
    ]
    lower = [-20.0, 0.0, 0.0, 1e-5, 0.0, 0.0, 0.0]
    upper = [20.0, 100.0, 100.0, 2.0, 1.0, 100.0, 10.0]
    best = None
    for x0 in starts:
        result = least_squares(lambda p: np.asarray([_predict_params(p, r) for r in train_rows]) - y,
                               x0=np.asarray(x0, dtype=float), bounds=(lower, upper), max_nfev=20000)
        loss = float(np.dot(result.fun, result.fun))
        if best is None or loss < best[0]:
            best = (loss, result.x)
    return best[1]

def predict(row):
    p = fit([row])
    return float(_predict_params(p, row))