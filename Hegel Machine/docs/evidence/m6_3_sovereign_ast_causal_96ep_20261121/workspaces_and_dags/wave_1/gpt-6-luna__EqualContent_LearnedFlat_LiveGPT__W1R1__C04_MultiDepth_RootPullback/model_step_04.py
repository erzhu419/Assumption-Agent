import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _predict_params(row, p):
    b, indoor_gain, attenuation, area_sensitivity, outdoor_gain = p
    indoor = float(row.get('u_indoor', 0.0))
    impact = float(row.get('impact_idx', 0.0))
    area = float(row.get('a_foot', 0.0))
    outdoor = float(row.get('u_outdoor', 0.0))
    weather = float(row.get('p_weather', 0.0))
    # Indoor exercise contributes vitality, attenuated by impact and constrained by area.
    effective_area = max(0.1, area)
    return b + indoor_gain * indoor * math.exp(-attenuation * impact * effective_area) + outdoor_gain * outdoor * weather

def fit(train_rows):
    global _PARAMS
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.array([_predict_params(r, p) for r in train_rows]) - y
    result = least_squares(residual, np.array([1.5, 1.5, 1.0, 0.0, 1.0]), max_nfev=10000)
    _PARAMS = result.x
    return predict

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return float(_predict_params(row, _PARAMS))