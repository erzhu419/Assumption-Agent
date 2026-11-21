import math
import numpy as np
from scipy.optimize import least_squares

_KEYS = ('u_outdoor', 'u_indoor', 'p_weather', 'a_foot', 'impact_idx')
_params = np.array([8.402, 1.169, 6.0, 1.5, 0.548])

def _law(params, outdoor, indoor, weather, footprint, impact):
    ceiling, half_dose, slope, midpoint, loss = params
    space = np.exp(-np.logaddexp(0.0, slope * (footprint - midpoint)))
    dose = weather * outdoor + space * indoor
    return ceiling * dose / (half_dose + dose) - loss * (indoor * impact)**2

def fit(train_rows):
    global _params
    data = np.asarray([[row[k] for k in (*_KEYS, 'y')] for row in train_rows], dtype=float).T
    x, y = data[:-1], data[-1]
    _params = least_squares(lambda p: _law(p, *x) - y, _params, bounds=([0, 1e-8, 0, -math.inf, 0], math.inf)).x
    return _params.copy()

def predict(row) -> float:
    return float(_law(_params, *(float(row[k]) for k in _KEYS)))