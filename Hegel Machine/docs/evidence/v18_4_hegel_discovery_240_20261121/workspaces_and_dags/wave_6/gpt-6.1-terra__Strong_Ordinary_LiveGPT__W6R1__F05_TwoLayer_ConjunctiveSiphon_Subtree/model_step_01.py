import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    def residual(p):
        vmax, rho, c_loc = p
        out = []
        for row in train_rows:
            u = row['u_now'] / 0.005
            s = row['s_grad']
            y = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + c_loc * math.sqrt(u)
            out.append(y - row['y'])
        return np.asarray(out)
    _params = least_squares(residual, [7.0, 0.7, -0.3], max_nfev=10000).x

def predict(row):
    vmax, rho, c_loc = _params
    u = row['u_now'] / 0.005
    s = row['s_grad']
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + c_loc * math.sqrt(u))