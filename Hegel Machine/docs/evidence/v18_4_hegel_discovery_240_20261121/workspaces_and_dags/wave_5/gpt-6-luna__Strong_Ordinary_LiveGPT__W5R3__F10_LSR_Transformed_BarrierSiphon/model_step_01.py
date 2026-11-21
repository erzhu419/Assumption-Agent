import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    tau_scale = 0.02
    def value(row, p):
        j0, beta_bar, rho, c_sqrt = p
        un = float(row['u_now']) / tau_scale
        sg = float(row['s_grad'])
        flux = j0 * un * math.exp(-beta_bar / (0.5 + un)) * (1.0 - rho * sg / (1.0 + sg))
        return flux + c_sqrt * math.sqrt(max(un, 0.0))
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([value(r, p) for r in train_rows]) - y
    starts = [[1.0, 0.5, 0.5, 0.0], [2.0, 1.0, 0.7, 0.1], [0.5, 0.1, 0.2, -0.1]]
    fits = [least_squares(residual, p, bounds=([0.0, 0.0, -5.0, -10.0], [100.0, 20.0, 5.0, 10.0]), max_nfev=20000) for p in starts]
    best = min(fits, key=lambda q: float(np.dot(q.fun, q.fun)))
    return {'params': best.x.tolist(), 'tau_scale': tau_scale}

def predict(row):
    p = _fit['params']
    un = float(row['u_now']) / _fit['tau_scale']
    sg = float(row['s_grad'])
    j0, beta_bar, rho, c_sqrt = p
    return float(j0 * un * math.exp(-beta_bar / (0.5 + un)) * (1.0 - rho * sg / (1.0 + sg)) + c_sqrt * math.sqrt(max(un, 0.0)))
