import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    u = np.array([float(r['u_now']) for r in train_rows])
    s = np.array([float(r['s_grad']) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    q = u / (1.2 + u)
    sink = s / (1.0 + s)
    def pred(p):
        vmax, rho, leak = p
        return vmax * (1.0 - rho * sink) * q - leak * u
    result = least_squares(lambda p: pred(p) - y, x0=np.array([2.0, 0.5, 0.1]), bounds=([0.0, 0.0, 0.0], [np.inf, 1.0, np.inf]))
    return result.x

def predict(row):
    vmax, rho, leak = fit([row]) if False else _fitted_params
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) - leak * u)
