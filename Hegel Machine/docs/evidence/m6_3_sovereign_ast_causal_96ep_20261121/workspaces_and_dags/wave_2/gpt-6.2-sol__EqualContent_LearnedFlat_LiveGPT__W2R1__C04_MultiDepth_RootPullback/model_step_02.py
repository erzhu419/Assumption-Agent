import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _features(row):
    return (float(row['u_outdoor']) * float(row['p_weather']),
            float(row['u_indoor']), float(row['a_foot']),
            float(row['impact_idx']))

def _law(params, x):
    amplitude, half_saturation, steepness, midpoint, loss = params
    outdoor, indoor, footprint, impact = x
    gate = 0.5 * (1.0 - math.tanh(0.5 * steepness * (footprint - midpoint)))
    effort = outdoor + indoor * gate
    return amplitude * effort / (half_saturation + effort) - loss * (indoor * impact)**2

def fit(train_rows):
    global _params
    samples = [(_features(row), float(row['y'])) for row in train_rows]
    def residuals(params):
        return np.array([_law(params, x) - y for x, y in samples])
    _params = least_squares(residuals, [9.205, 1.447, 6.0, 1.5, 0.473],
                           bounds=([0, 1e-8, 0, 0, 0], np.inf)).x

def predict(row) -> float:
    return float(_law(_params, _features(row)))