import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _features(row):
    indoor = float(row['u_indoor'])
    t = 6.0 * (float(row['a_foot']) - 1.5)
    q = math.exp(-abs(t))
    adherence = q / (1.0 + q) if t >= 0 else 1.0 / (1.0 + q)
    dose = float(row['u_outdoor']) * float(row['p_weather']) + indoor * adherence
    impact = (indoor * float(row['impact_idx'])) ** 2
    return dose, impact

def fit(train_rows):
    global _params
    rows = list(train_rows)
    dose, impact = np.asarray([_features(row) for row in rows], dtype=float).T
    y = np.asarray([float(row['y']) for row in rows])

    def residual(p):
        gain, half_dose, loss = p
        return gain * dose / (half_dose + dose) - loss * impact - y

    _params = least_squares(residual, [8.0, 1.5, 0.4], bounds=([0.0, 1e-9, 0.0], np.inf), ftol=1e-12, xtol=1e-12, gtol=1e-12).x

def predict(row) -> float:
    gain, half_dose, loss = _params
    dose, impact = _features(row)
    return float(gain * dose / (half_dose + dose) - loss * impact)
