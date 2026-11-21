import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    def pred_params(p, row):
        vmax, rho, c_loc = p
        u = max(0.0, float(row['u_now']))
        s = float(row['s_grad'])
        base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
        return base + c_loc * math.sqrt(u)
    y = np.asarray([float(r['y']) for r in rows])
    def residual(p):
        return np.asarray([pred_params(p, r) for r in rows]) - y
    result = least_squares(residual, x0=np.array([4.0, 0.7, 0.2]), bounds=([0.0, 0.0, -10.0], [20.0, 1.5, 10.0]), max_nfev=10000)
    return result.x.tolist()

def predict(row):
    p = fit([row])
    return float(p[0] * (1.0 - p[1] * float(row['s_grad']) / (1.0 + float(row['s_grad']))) * (float(row['u_now']) / (1.2 + float(row['u_now']))) + p[2] * math.sqrt(max(0.0, float(row['u_now']))))