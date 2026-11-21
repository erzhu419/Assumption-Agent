import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = train_rows
    def predict_params(p, row):
        amp, log_tau = p
        tau = math.exp(log_tau)
        denom = row['u_now'] + row['u_past']
        z = (row['u_now'] - row['u_past']) / denom if abs(denom) > 1e-15 else 0.0
        return row['x_now'] - amp * z * math.exp(-max(0.0, row['H']) / tau)
    def residual(p):
        return np.array([predict_params(p, r) - r['y'] for r in rows])
    p0 = np.array([1.0, math.log(0.5)])
    result = least_squares(residual, p0, max_nfev=20000)
    return result.x

def predict(row):
    p = fit(_TRAIN_ROWS)
    denom = row['u_now'] + row['u_past']
    z = (row['u_now'] - row['u_past']) / denom if abs(denom) > 1e-15 else 0.0
    return row['x_now'] - p[0] * z * math.exp(-max(0.0, row['H']) / math.exp(p[1]))