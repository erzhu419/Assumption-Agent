import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, params):
    gain, half_dose, midpoint, steepness, loss = params
    indoor = float(row['u_indoor'])
    footprint = float(row['a_foot'])
    impact = float(row['impact_idx'])
    space = 0.5 * (1.0 - math.tanh(0.5 * steepness * (footprint - midpoint)))
    dose = float(row['u_outdoor']) * float(row['p_weather']) + indoor * space
    return gain * dose / (half_dose + dose) - loss * (indoor * impact)**2

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.array([float(row['y']) for row in rows])
    result = least_squares(
        lambda p: np.array([_value(row, p) for row in rows]) - y,
        x0=[8.4, 1.17, 1.5, 6.0, 0.55],
        bounds=([0.0, 1e-9, 0.0, 0.0, 0.0], np.inf),
    )
    _params = result.x

def predict(row):
    return float(_value(row, _params))