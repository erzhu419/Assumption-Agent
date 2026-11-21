import math
import numpy as np
from scipy.optimize import least_squares

_params = None
_keys = ('u_outdoor', 'u_indoor', 'p_weather', 'a_foot', 'impact_idx')

def fit(train_rows):
    global _params
    outdoor, indoor, weather, footprint, impact = np.array([[r[k] for k in _keys] for r in train_rows], dtype=float).T
    y = np.array([r['y'] for r in train_rows], dtype=float)
    outdoor_dose = outdoor * weather
    impact_load_sq = (indoor * impact) ** 2

    def residual(params):
        scale, half_sat, slope, midpoint, loss = params
        gate = 0.5 * (1.0 + np.tanh(0.5 * slope * (midpoint - footprint)))
        dose = outdoor_dose + indoor * gate
        return scale * dose / (half_sat + dose) - loss * impact_load_sq - y

    _params = least_squares(residual, [8.4, 1.17, 6.0, 1.5, 0.55], bounds=([0.0, 1e-8, 0.0, 0.0, 0.0], np.inf)).x

def predict(row) -> float:
    scale, half_sat, slope, midpoint, loss = _params
    outdoor, indoor, weather, footprint, impact = (float(row[k]) for k in _keys)
    gate = 0.5 * (1.0 + math.tanh(0.5 * slope * (midpoint - footprint)))
    dose = outdoor * weather + indoor * gate
    return float(scale * dose / (half_sat + dose) - loss * (indoor * impact) ** 2)
