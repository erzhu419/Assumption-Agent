import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    x = np.array([float(r['x_ind']) for r in train_rows])
    u = np.array([float(r['u_ctrl']) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        vmax, k, c = p
        return vmax*x/(k+x) + c*u - y
    sol = least_squares(residual, x0=[8.0, 1.0, 0.5], bounds=([0.0, 1e-8, -np.inf], [np.inf, np.inf, np.inf]))
    return sol.x

def predict(row):
    vmax, k, c = fit(_MODEL_TRAIN_ROWS)
    x = float(row['x_ind']); u = float(row['u_ctrl'])
    return float(vmax*x/(k+x) + c*u)