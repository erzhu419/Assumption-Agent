import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    z = np.asarray([float(r['z_2']) for r in train_rows], dtype=float)
    u = np.asarray([float(r['u_ctrl']) for r in train_rows], dtype=float)
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        vmax, k_half, c_ctrl = p
        return vmax * z**2 / (k_half + z**2) - c_ctrl * u - y
    initial = [max(float(np.max(y)), 1.0), 1.0, 0.1]
    result = least_squares(residual, initial, bounds=([0.0, 1e-12, 0.0], [np.inf, np.inf, np.inf]))
    return tuple(float(v) for v in result.x)

def predict(row):
    vmax, k_half, c_ctrl = fit(_fit_rows)
    z = float(row['z_2'])
    u = float(row['u_ctrl'])
    return float(vmax * z**2 / (k_half + z**2) - c_ctrl * u)
