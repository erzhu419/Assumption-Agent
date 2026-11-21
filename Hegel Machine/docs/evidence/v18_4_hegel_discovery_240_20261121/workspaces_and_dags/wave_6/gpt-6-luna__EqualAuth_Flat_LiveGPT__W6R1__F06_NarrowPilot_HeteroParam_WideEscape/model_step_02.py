import math
import numpy as np
from scipy.optimize import least_squares

def _model(p, row):
    vmax, rho, c_loc, k_aval, s_crit = p
    u = max(1e-4, float(row['u_now']))
    s = float(row['s_grad'])
    return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_loc * math.sqrt(u) - k_aval * max(0.0, s - s_crit) * u

def fit(train_rows):
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_model(p, r) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([5.0, 0.4, 0.5, 0.5, 2.6]), bounds=([0.0, 0.0, 0.0, 0.0, 2.0], [50.0, 1.0, 10.0, 20.0, 3.5]), max_nfev=20000)
    return result.x.tolist()

def predict(row):
    p = fit(row['train_rows']) if 'train_rows' in row else None
    if p is None:
        raise ValueError('predict expects row with fitted parameters as _params')
    return float(_model(p, row))