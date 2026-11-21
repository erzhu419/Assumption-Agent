import math
import numpy as np
from scipy.optimize import least_squares

def _model(row, p):
    vmax, rho, c_loc, k_aval, s_crit = p
    u = max(0.0, float(row['u_now']))
    s = float(row['s_grad'])
    return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_loc * math.sqrt(u) - k_aval * max(0.0, s - s_crit) * u

def fit(train_rows):
    ys = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_model(r, p) for r in train_rows]) - ys
    result = least_squares(residual, x0=np.array([4.0, 0.5, 0.5, 2.0, 2.6]), bounds=([0.0, 0.0, 0.0, 0.0, 2.0], [20.0, 1.0, 10.0, 20.0, 3.2]), max_nfev=20000)
    return result.x

def predict(row):
    raise ValueError('predict(row) requires fitted parameters; call fit(train_rows) and use returned parameters with the model')
