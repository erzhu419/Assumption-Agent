import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    u = np.array([max(1e-9, float(r['u_now'])) for r in rows])
    s = np.array([max(0.0, float(r['s_grad'])) for r in rows])
    y = np.array([float(r['y']) for r in rows])
    def pred(p, uu, ss):
        vmax, rho, c_loc = p
        return vmax * (1.0 - rho * ss / (1.0 + ss)) * (uu / (1.2 + uu)) + c_loc * np.sqrt(uu)
    res = least_squares(lambda p: pred(p, u, s) - y, x0=np.array([5.0, 0.7, 0.5]), bounds=([0.0, 0.0, -20.0], [100.0, 1.5, 20.0]), max_nfev=20000)
    return res.x

def predict(row):
    p = fit(_TRAIN_ROWS)
    u = max(1e-9, float(row['u_now']))
    s = max(0.0, float(row['s_grad']))
    vmax, rho, c_loc = p
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_loc * math.sqrt(u))