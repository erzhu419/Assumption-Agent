import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _features(row):
    indoor = float(row['u_indoor'])
    dose = float(row['u_outdoor']) * float(row['p_weather'])
    dose += indoor / (1.0 + math.exp(6.0 * (float(row['a_foot']) - 1.5)))
    penalty = (indoor * float(row['impact_idx'])) ** 2
    return dose, penalty

def fit(train_rows):
    global _params
    rows = list(train_rows)
    dose, penalty = np.asarray([_features(row) for row in rows], dtype=float).T
    target = np.asarray([row['y'] for row in rows], dtype=float)

    def residual(params):
        gain, half_dose, impact_cost = params
        return gain * dose / (half_dose + dose) - impact_cost * penalty - target

    _params = least_squares(residual, [8.0, 1.0, 0.5], bounds=([0.0, 1e-9, 0.0], [np.inf, np.inf, np.inf])).x

def predict(row) -> float:
    gain, half_dose, impact_cost = _params
    dose, penalty = _features(row)
    return float(gain * dose / (half_dose + dose) - impact_cost * penalty)
