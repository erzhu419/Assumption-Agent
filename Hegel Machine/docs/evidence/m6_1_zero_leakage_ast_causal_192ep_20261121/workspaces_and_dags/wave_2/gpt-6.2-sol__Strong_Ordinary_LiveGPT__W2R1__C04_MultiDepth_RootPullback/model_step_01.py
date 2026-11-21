import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _features(row):
    indoor = float(row['u_indoor'])
    z = 6.0 * (float(row['a_foot']) - 1.5)
    space = math.exp(-float(np.logaddexp(0.0, z)))
    dose = float(row['u_outdoor']) * float(row['p_weather']) + indoor * space
    impact_sq = (indoor * float(row['impact_idx'])) ** 2
    return dose, impact_sq

def fit(train_rows):
    global _params
    rows = list(train_rows)
    dose, impact_sq = np.array([_features(row) for row in rows], dtype=float).T
    y = np.array([row['y'] for row in rows], dtype=float)

    def residual(p):
        gain, half_dose, impact_cost = p
        return gain * dose / (half_dose + dose) - impact_cost * impact_sq - y

    _params = least_squares(residual, [8.0, 1.0, 0.5], bounds=([0.0, 1e-8, 0.0], np.inf)).x

def predict(row) -> float:
    gain, half_dose, impact_cost = _params
    dose, impact_sq = _features(row)
    return float(gain * dose / (half_dose + dose) - impact_cost * impact_sq)
