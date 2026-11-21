import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    u = np.array([max(1e-9, float(r['u_now'])) for r in rows])
    s = np.array([float(r['s_grad']) for r in rows])
    y = np.array([float(r['y']) for r in rows])
    def pred(p):
        vmax, rho, c_sqrt = p
        return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_sqrt * np.sqrt(u)
    result = least_squares(lambda p: pred(p) - y, x0=np.array([5.0, 0.5, 1.0]), bounds=([0.0, 0.0, -10.0], [100.0, 5.0, 10.0]))
    return result.x

def predict(row):
    vmax, rho, c_sqrt = fit(_active_train_rows)
    u = max(1e-9, float(row['u_now']))
    s = float(row['s_grad'])
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_sqrt * math.sqrt(u))