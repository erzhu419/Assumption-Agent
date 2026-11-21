import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _features(row):
    indoor = float(row['u_indoor'])
    outdoor_dose = float(row['u_outdoor']) * float(row['p_weather'])
    impact_load = float(row['a_foot']) * float(row['impact_idx'])
    return indoor, outdoor_dose, impact_load

def _predict_params(p, row):
    indoor, outdoor_dose, impact_load = _features(row)
    baseline, amplitude, half_sat, outdoor_gain, impact_loss = p
    dose = indoor / (half_sat + indoor) if indoor >= 0 else indoor / (half_sat + indoor)
    return baseline + amplitude * dose + outdoor_gain * outdoor_dose - impact_loss * impact_load

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows], dtype=float)
    x0 = np.array([1.0, 8.0, 1.5, 1.0, 3.0])
    lo = np.array([-20.0, 0.001, 0.001, -20.0, 0.0])
    hi = np.array([20.0, 30.0, 20.0, 20.0, 30.0])
    result = least_squares(lambda p: np.asarray([_predict_params(p, r) for r in rows]) - y, x0=x0, bounds=(lo, hi), max_nfev=20000)
    _params = result.x
    return _params.copy()

def predict(row):
    if _params is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    return float(_predict_params(_params, row))