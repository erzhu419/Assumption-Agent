import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    u = np.array([float(r['u_now']) for r in train_rows])
    s = np.array([float(r['s_grad']) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    def pred(p):
        vmax, rho, c_loc = p
        return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_loc * np.sqrt(u)
    res = least_squares(lambda p: pred(p) - y, x0=np.array([6.0, 0.5, 0.3]), bounds=([0.0, 0.0, -10.0], [100.0, 1.0, 10.0]))
    return tuple(float(v) for v in res.x)

def predict(row):
    vmax, rho, c_loc = fit(_train_rows)
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_loc * math.sqrt(u))
