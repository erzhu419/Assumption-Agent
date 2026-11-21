import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    def residual(p):
        vmax, rho, c = p
        return np.array([vmax * (1.0 - rho * float(r['s_grad']) / (1.0 + float(r['s_grad']))) * (float(r['u_now']) / (1.2 + float(r['u_now']))) + c * math.sqrt(float(r['u_now'])) - float(r['y']) for r in train_rows])
    result = least_squares(residual, [10.0, 0.7, 1.0], max_nfev=10000)
    _params = result.x

def predict(row):
    vmax, rho, c = _params
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + c * math.sqrt(u))