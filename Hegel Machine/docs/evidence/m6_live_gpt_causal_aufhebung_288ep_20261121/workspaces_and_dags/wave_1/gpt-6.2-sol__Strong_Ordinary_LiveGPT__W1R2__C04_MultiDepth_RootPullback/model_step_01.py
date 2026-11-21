import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _features(row):
    indoor = row['u_indoor']
    adherence = 1.0 / (1.0 + math.exp(6.0 * (row['a_foot'] - 1.5)))
    dose = row['u_outdoor'] * row['p_weather'] + indoor * adherence
    impact = (indoor * row['impact_idx']) ** 2
    return dose, impact

def fit(train_rows):
    global _params
    rows = list(train_rows)
    dose, impact = np.asarray([_features(row) for row in rows], dtype=float).T
    y = np.asarray([row['y'] for row in rows], dtype=float)

    def residual(params):
        gain, half_dose, penalty = params
        return gain * dose / (half_dose + dose) - penalty * impact - y

    _params = least_squares(residual, [8.0, 1.5, 0.4], bounds=([0.0, 1e-9, 0.0], [np.inf, np.inf, np.inf])).x

def predict(row) -> float:
    gain, half_dose, penalty = _params
    dose, impact = _features(row)
    return float(gain * dose / (half_dose + dose) - penalty * impact)
