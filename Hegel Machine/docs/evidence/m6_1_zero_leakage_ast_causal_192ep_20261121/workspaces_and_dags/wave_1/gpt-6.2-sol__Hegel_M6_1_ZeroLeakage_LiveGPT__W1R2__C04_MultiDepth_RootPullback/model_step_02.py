import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _features(row):
    indoor = float(row['u_indoor'])
    z = 6.0 * (float(row['a_foot']) - 1.5)
    t = math.exp(-abs(z))
    envelope = t / (1.0 + t) if z >= 0.0 else 1.0 / (1.0 + t)
    dose = float(row['u_outdoor']) * float(row['p_weather']) + indoor * envelope
    impact_sq = (indoor * float(row['impact_idx'])) ** 2
    return dose, impact_sq

def fit(train_rows):
    global _params
    rows = list(train_rows)
    dose, impact_sq = np.asarray([_features(row) for row in rows], dtype=float).T
    y = np.asarray([float(row['y']) for row in rows], dtype=float)

    def residual(p):
        gain, half_dose, impact_cost = p
        return gain * dose / (half_dose + dose) - impact_cost * impact_sq - y

    initial = [max(1.0, 1.5 * float(np.max(y))), max(0.1, float(np.median(dose))), 0.1]
    _params = least_squares(residual, initial, bounds=([0.0, 1e-9, 0.0], np.inf)).x

def predict(row):
    dose, impact_sq = _features(row)
    gain, half_dose, impact_cost = _params
    return float(gain * dose / (half_dose + dose) - impact_cost * impact_sq)
