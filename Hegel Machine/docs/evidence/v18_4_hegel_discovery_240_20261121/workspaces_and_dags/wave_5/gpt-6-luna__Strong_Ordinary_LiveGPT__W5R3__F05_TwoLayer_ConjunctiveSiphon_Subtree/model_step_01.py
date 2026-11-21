import math
import numpy as np
from scipy.optimize import least_squares

def _features(row):
    u = float(row['u_now']) / 4.0
    s = float(row['s_grad'])
    base = (u / (1.2 + u))
    sink = 1.0 - s / (1.0 + s)
    return u, base * sink

def fit(train_rows):
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        vmax, rho, c_loc = p
        pred = []
        for r in rows:
            u, b = _features(r)
            pred.append(vmax * (1.0 - rho * float(r['s_grad']) / (1.0 + float(r['s_grad']))) * (u / (1.2 + u)) + c_loc * math.sqrt(max(u, 0.0)))
        return np.asarray(pred) - y
    result = least_squares(residual, x0=np.array([5.0, 0.65, 0.0]), bounds=([0.0, 0.0, -10.0], [100.0, 1.0, 10.0]))
    return result.x

def predict(row):
    vmax, rho, c_loc = fit(_training_rows)
    u, b = _features(row)
    s = float(row['s_grad'])
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_loc * math.sqrt(max(u, 0.0)))