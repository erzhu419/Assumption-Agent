import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _predict_with(row, p):
    base, b_out, b_in, k = p
    outdoor = float(row.get('u_outdoor', 0.0)) * float(row.get('p_weather', 0.0))
    indoor = float(row.get('u_indoor', 0.0))
    impact_area = float(row.get('impact_idx', 0.0)) * float(row.get('a_foot', 0.0))
    return (base + b_out * outdoor + b_in * indoor) * math.exp(-k * impact_area)

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([_predict_with(r, p) for r in rows]) - y
    result = least_squares(residual, x0=np.array([1.0, 1.0, 1.5, 1.0]), bounds=([-100.0, -100.0, -100.0, 0.0], [100.0, 100.0, 100.0, 20.0]))
    _PARAMS = result.x.copy()
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return float(_predict_with(row, _PARAMS))