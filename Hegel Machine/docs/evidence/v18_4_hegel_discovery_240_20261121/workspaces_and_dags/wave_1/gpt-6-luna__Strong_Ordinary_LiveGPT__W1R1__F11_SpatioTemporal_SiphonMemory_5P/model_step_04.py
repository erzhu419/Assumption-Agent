import math
import numpy as np
from scipy.optimize import least_squares

def _pred(row, p):
    j0, beta_bar, rho, c_sqrt = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    flux = j0 * u * math.exp(-beta_bar / (0.5 + u)) * (1.0 - rho * s / (1.0 + s))
    return flux + c_sqrt * math.sqrt(max(u, 0.0))

def fit(train_rows):
    def residual(p):
        return np.array([_pred(r, p) - float(r['y']) for r in train_rows])
    result = least_squares(residual, x0=np.array([2.0, 0.5, 0.7, 0.1]), max_nfev=20000)
    return result.x

def predict(row):
    return float(_pred(row, fit([dict(row, y=0.0)])))