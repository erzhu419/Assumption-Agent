import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    def pred(p, r):
        gain, tau_relax, k_eq = p
        tau_relax = max(tau_relax, 1e-8)
        un = float(r['u_now'])
        up = float(r['u_past'])
        H = float(r['H'])
        return gain * (un - up) * math.exp(-H / tau_relax) + k_eq * un
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([pred(p, r) for r in train_rows]) - y
    best = None
    for start in ([0.5, 0.5, 0.5], [1.0, 1.0, 1.0], [-0.5, 1.0, 1.0]):
        res = least_squares(residual, np.array(start, dtype=float), bounds=([-20.0, 0.01, -20.0], [20.0, 20.0, 20.0]))
        if best is None or np.sum(res.fun ** 2) < np.sum(best.fun ** 2):
            best = res
    return best.x

def predict(row):
    p = fit(row.get('train_rows', []))
    return float(p[0] * (float(row['u_now']) - float(row['u_past'])) * math.exp(-float(row['H']) / max(float(p[1]), 1e-8)) + p[2] * float(row['u_now']))