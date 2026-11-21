import math
import numpy as np
from scipy.optimize import least_squares

params = None

def fit(train_rows):
    global params
    def residual(p):
        vmax, rho, c_sqrt = p
        return np.array([vmax * (1.0 - rho * r['s_grad'] / (1.0 + r['s_grad'])) * r['u_now'] / (1.2 + r['u_now']) + c_sqrt * math.sqrt(r['u_now']) - r['y'] for r in train_rows])
    params = least_squares(residual, [8.0, 0.7, 0.5]).x

def predict(row):
    vmax, rho, c_sqrt = params
    u = row['u_now']
    s = row['s_grad']
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + c_sqrt * math.sqrt(u))
