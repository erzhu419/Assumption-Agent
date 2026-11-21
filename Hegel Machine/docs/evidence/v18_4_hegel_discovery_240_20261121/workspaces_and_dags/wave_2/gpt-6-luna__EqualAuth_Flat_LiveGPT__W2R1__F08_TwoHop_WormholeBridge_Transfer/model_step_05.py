import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = train_rows
    u = np.array([max(1e-9, float(r['u_now'])) for r in rows])
    s = np.array([float(r['s_grad']) for r in rows])
    y = np.array([float(r['y']) for r in rows])
    base = (u / (1.2 + u))
    def calc(p, sign):
        vmax, rho, c = p
        return vmax * (1.0 - rho * s / (1.0 + s)) * base + sign * c * np.sqrt(u)
    best = None
    for sign in (1.0, -1.0):
        res = least_squares(lambda p: calc(p, sign) - y, x0=np.array([5.0, 0.7, 0.1]), bounds=([0.0, 0.0, 0.0], [100.0, 5.0, 100.0]))
        err = float(np.mean((calc(res.x, sign) - y) ** 2))
        if best is None or err < best[0]:
            best = (err, sign, res.x)
    return (float(best[1]),) + tuple(float(v) for v in best[2])

def predict(row):
    sign, vmax, rho, c = fit(train_rows)
    u = max(1e-9, float(row['u_now']))
    s = float(row['s_grad'])
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + sign * c * math.sqrt(u))