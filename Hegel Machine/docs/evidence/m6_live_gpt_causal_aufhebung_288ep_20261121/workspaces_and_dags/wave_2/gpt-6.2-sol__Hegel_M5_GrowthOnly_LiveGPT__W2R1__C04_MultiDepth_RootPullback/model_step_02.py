import math
import numpy as np
from scipy.optimize import least_squares

_params = None
_keys = ('u_outdoor', 'u_indoor', 'p_weather', 'a_foot', 'impact_idx')

def _model(params, outdoor, indoor, weather, footprint, impact):
    gain, half_sat, loss, slope, midpoint = params
    envelope = np.exp(-np.logaddexp(0.0, slope * (footprint - midpoint)))
    dose = outdoor * weather + indoor * envelope
    return gain * dose / (half_sat + dose) - loss * (indoor * impact)**2

def fit(train_rows):
    global _params
    rows = list(train_rows)
    x = np.array([[r[k] for k in _keys] for r in rows], dtype=float).T
    y = np.array([r['y'] for r in rows], dtype=float)
    result = least_squares(lambda p: _model(p, *x) - y,
                           [9.0, 1.5, 0.5, 6.0, 1.5],
                           bounds=([0.0, 1e-8, 0.0, 0.0, 0.0], np.inf),
                           ftol=1e-12, xtol=1e-12, gtol=1e-12)
    _params = result.x
    return _params

def predict(row) -> float:
    return float(_model(_params, *(float(row[k]) for k in _keys)))