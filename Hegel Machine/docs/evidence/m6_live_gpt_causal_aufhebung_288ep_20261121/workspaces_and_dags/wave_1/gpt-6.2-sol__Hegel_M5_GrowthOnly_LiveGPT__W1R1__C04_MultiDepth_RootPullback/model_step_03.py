import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _law(row, params):
    gain, half_dose, midpoint, slope, loss = params
    indoor = float(row['u_indoor'])
    z = slope * (float(row['a_foot']) - midpoint)
    q = math.exp(-abs(z))
    space_factor = q / (1.0 + q) if z >= 0.0 else 1.0 / (1.0 + q)
    dose = float(row['u_outdoor']) * float(row['p_weather']) + indoor * space_factor
    return gain * dose / (half_dose + dose) - loss * (indoor * float(row['impact_idx'])) ** 2

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.array([float(row['y']) for row in rows])
    def residual(params):
        return np.array([_law(row, params) for row in rows]) - y
    _params = least_squares(residual, [8.255, 0.874, 1.5, 6.0, 0.47], bounds=([0.0, 1e-8, 0.0, 0.0, 0.0], np.inf)).x

def predict(row):
    return float(_law(row, _params))