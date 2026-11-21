import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    u = np.array([max(float(r['u_now']), 0.0) for r in rows])
    s = np.array([float(r['s_grad']) for r in rows])
    def pred(p):
        vmax, rho, c_loc = p
        return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_loc * np.sqrt(u)
    def residual(p):
        return pred(p) - y
    result = least_squares(residual, x0=[6.0, 0.7, 0.1], bounds=([1.0, 0.05, -2.0], [15.0, 0.95, 2.0]), max_nfev=20000)
    return result.x

def predict(row):
    vmax, rho, c_loc = fit([row])
    u = max(float(row['u_now']), 0.0)
    s = float(row['s_grad'])
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_loc * np.sqrt(u))