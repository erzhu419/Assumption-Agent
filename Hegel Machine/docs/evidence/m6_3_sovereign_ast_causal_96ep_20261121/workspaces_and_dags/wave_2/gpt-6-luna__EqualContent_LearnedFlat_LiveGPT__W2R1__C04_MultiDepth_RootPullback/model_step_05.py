import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(p, row):
    outdoor = float(row['u_outdoor'])
    indoor = float(row['u_indoor'])
    weather = float(row['p_weather'])
    area = float(row['a_foot'])
    impact = float(row['impact_idx'])
    # Saturating dose benefit; footprint and impact attenuate the benefit.
    dose = p[1] * indoor + p[2] * outdoor * weather
    attenuation = math.exp(-max(-30.0, min(30.0, p[3] * impact * area)))
    return p[0] + p[4] * (1.0 - math.exp(-max(0.0, dose))) * attenuation

def fit(train_rows):
    global _params
    ys = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_value(p, r) for r in train_rows]) - ys
    result = least_squares(residual, np.array([0.0, 0.7, 1.0, 1.0, 6.0]), max_nfev=20000)
    _params = result.x
    return predict

def predict(row):
    if _params is None:
        raise RuntimeError('fit must be called before predict')
    return float(_value(_params, row))