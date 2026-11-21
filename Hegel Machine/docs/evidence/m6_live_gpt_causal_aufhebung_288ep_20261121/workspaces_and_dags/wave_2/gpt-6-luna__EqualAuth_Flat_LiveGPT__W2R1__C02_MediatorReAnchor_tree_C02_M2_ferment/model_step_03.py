import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    s = np.array([r['s_surf'] for r in train_rows], dtype=float)
    u = np.array([r['u_raw'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        vmax, km, c = p
        return vmax*s/(km+s) + c*np.sqrt(np.maximum(u, 0)) - y
    result = least_squares(residual, x0=[8.0, 1.0, 0.2], bounds=([0, 1e-8, -np.inf], [np.inf, np.inf, np.inf]))
    params = result.x
    def predict(row):
        vmax, km, c = params
        return float(vmax*row['s_surf']/(km+row['s_surf']) + c*math.sqrt(max(row['u_raw'], 0.0)))
    return predict

def predict(row):
    raise RuntimeError('Call fit(train_rows) first and use its returned predictor.')