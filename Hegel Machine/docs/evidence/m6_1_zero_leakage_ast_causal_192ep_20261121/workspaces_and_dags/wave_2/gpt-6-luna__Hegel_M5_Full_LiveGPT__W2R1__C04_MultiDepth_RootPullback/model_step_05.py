import math
import numpy as np
from scipy.optimize import least_squares

def _features(row):
    dose = float(row['u_indoor']) + float(row['u_outdoor']) * float(row['p_weather'])
    burden = float(row['impact_idx']) * float(row['a_foot'])
    return dose, burden

def _law(params, row):
    baseline, gain, half_sat, attenuation = params
    dose, burden = _features(row)
    return baseline + gain * dose / (half_sat + max(dose, 0.0)) * math.exp(-attenuation * burden)

def fit(train_rows):
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_law(p, r) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([0.0, 8.0, 1.0, 1.0]),
                           bounds=([-20.0, 0.0, 1e-6, 0.0], [20.0, 30.0, 30.0, 20.0]),
                           max_nfev=20000)
    return result.x

def predict(row):
    return float(_law(fit(_TRAIN_ROWS), row))