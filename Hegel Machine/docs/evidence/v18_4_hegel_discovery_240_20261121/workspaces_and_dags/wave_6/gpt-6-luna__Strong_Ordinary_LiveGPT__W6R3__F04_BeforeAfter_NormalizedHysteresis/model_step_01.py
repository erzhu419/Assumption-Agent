import math
import numpy as np
from scipy.optimize import least_squares

TAU_SCALE = 0.02

def _features(row):
    un = float(row['u_now']) / TAU_SCALE
    up = float(row['u_past']) / TAU_SCALE
    return un, up, float(row['H'])

def fit(train_rows):
    def predict_params(p, row):
        gain, tau_relax, k_eq = p
        un, up, H = _features(row)
        return gain * (un - up) * math.exp(-H / tau_relax) + k_eq * un
    def residuals(p):
        return [predict_params(p, r) - float(r['y']) for r in train_rows]
    result = least_squares(residuals, x0=np.array([1.0, 0.1, 0.0]), bounds=([-np.inf, 1e-8, -np.inf], [np.inf, np.inf, np.inf]))
    return {'gain': float(result.x[0]), 'tau_relax': float(result.x[1]), 'k_eq': float(result.x[2])}

def predict(row):
    p = fit(_TRAIN_ROWS)
    un, up, H = _features(row)
    return float(p['gain'] * (un - up) * math.exp(-H / p['tau_relax']) + p['k_eq'] * un)