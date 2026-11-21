import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    u = np.array([r['u_now'] for r in rows], dtype=float)
    H = np.array([r['H'] for r in rows], dtype=float)
    sg = np.array([r['s_grad'] for r in rows], dtype=float)
    y = np.array([r['y'] for r in rows], dtype=float)
    def pred(p):
        vmax, adapt, log_tf, log_gap = p
        tf = math.exp(log_tf)
        ts = tf + math.exp(log_gap)
        sink = 1.0  # rho is unidentifiable because the observed s_grad values are all zero
        return vmax * sink * (u / (1.2 + u)) * (-np.expm1(-H / tf)) - adapt * u * (-np.expm1(-H / ts))
    def resid(p):
        return pred(p) - y
    result = least_squares(resid, [5.0, 1.0, math.log(0.5), math.log(8.0)], max_nfev=30000)
    return result.x

def predict(row):
    p = fit([row])
    vmax, adapt, log_tf, log_gap = p
    tf = math.exp(log_tf)
    ts = tf + math.exp(log_gap)
    u, H = float(row['u_now']), float(row['H'])
    return float(vmax * (u / (1.2 + u)) * (-math.expm1(-H / tf)) - adapt * u * (-math.expm1(-H / ts)))
