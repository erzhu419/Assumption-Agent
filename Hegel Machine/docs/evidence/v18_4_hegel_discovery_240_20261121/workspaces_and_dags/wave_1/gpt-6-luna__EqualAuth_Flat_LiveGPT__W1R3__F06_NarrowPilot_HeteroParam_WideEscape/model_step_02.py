import numpy as np
from scipy.optimize import least_squares

def _pred(p, row):
    vmax, rho = p
    u = max(0.0, float(row['u_now']))
    s = float(row['s_grad'])
    return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))

def fit(train_rows):
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_pred(p, r) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([8.0, 0.70]), bounds=([0.0, 0.0], [100.0, 0.999]))
    return result.x

def predict(row):
    p = fit(_TRAIN_ROWS)
    return float(_pred(p, row))
