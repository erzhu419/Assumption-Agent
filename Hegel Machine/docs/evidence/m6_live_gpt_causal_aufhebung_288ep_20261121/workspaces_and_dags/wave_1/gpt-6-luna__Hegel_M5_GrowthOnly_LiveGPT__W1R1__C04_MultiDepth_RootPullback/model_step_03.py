import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _value(row, name):
    return float(row[name])

def _predict_with(row, params):
    gain, half_sat, loss_rate = params
    effective_dose = _value(row, 'u_indoor') + _value(row, 'u_outdoor') * _value(row, 'p_weather')
    impact_loss = loss_rate * _value(row, 'impact_idx') * _value(row, 'a_foot')
    return gain * effective_dose / (half_sat + effective_dose) - impact_loss

def fit(train_rows):
    global _PARAMS
    y = np.asarray([_value(r, 'y') for r in train_rows], dtype=float)
    def residual(params):
        return np.asarray([_predict_with(r, params) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([8.0, 1.0, 3.0]), bounds=([0.0, 1e-8, 0.0], [100.0, 100.0, 100.0]))
    _PARAMS = result.x
    return _PARAMS.copy()

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    return float(_predict_with(row, _PARAMS))