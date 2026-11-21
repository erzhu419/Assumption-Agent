import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    u = np.array([max(1e-12, float(r['u_now'])) for r in train_rows])
    s = np.array([float(r['s_grad']) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    def predict_params(p, u, s):
        vmax, rho, c_loc = p
        return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_loc * np.sqrt(u)
    def residual(p):
        return predict_params(p, u, s) - y
    result = least_squares(residual, x0=np.array([6.5, 0.70, 0.1]), bounds=([1.0, 0.05, -2.0], [15.0, 0.95, 2.0]))
    return result.x

def predict(row):
    vmax, rho, c_loc = fit([row])
    u = max(1e-12, float(row['u_now']))
    s = float(row['s_grad'])
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_loc * np.sqrt(u))