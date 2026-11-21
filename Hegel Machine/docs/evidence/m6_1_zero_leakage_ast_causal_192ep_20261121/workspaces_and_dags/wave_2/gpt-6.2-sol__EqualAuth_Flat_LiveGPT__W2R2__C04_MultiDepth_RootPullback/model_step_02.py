import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _law(row, params):
    scale, half_sat, steepness, midpoint, loss = params
    indoor = float(row['u_indoor'])
    z = steepness * (float(row['a_foot']) - midpoint)
    gate = math.exp(-float(np.logaddexp(0.0, z)))
    effort = float(row['u_outdoor']) * float(row['p_weather']) + indoor * gate
    return float(scale * effort / (half_sat + effort) - loss * (indoor * float(row['impact_idx'])) ** 2)

def fit(train_rows):
    global _params
    rows = list(train_rows)
    def residuals(params):
        return np.array([_law(row, params) - float(row['y']) for row in rows])
    _params = least_squares(residuals, [8.0, 1.0, 6.0, 1.5, 0.5], bounds=([0.0, 1e-8, 0.0, 0.0, 0.0], np.inf)).x

def predict(row) -> float:
    return _law(row, _params)
