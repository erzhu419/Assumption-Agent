import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows])
    dose = np.asarray([float(r['u_indoor']) + float(r['u_outdoor']) * float(r['p_weather']) for r in rows])
    impact = np.asarray([float(r['impact_idx']) * float(r['a_foot']) for r in rows])
    def residual(q):
        gain, half_sat, loss = q
        pred = gain * dose / (half_sat + dose) - loss * impact
        return pred - y
    result = least_squares(residual, x0=np.array([7.0, 0.5, 4.0]), bounds=([0.0, 1e-8, 0.0], [100.0, 100.0, 100.0]))
    _PARAMS = tuple(float(v) for v in result.x)
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    gain, half_sat, loss = _PARAMS
    dose = float(row['u_indoor']) + float(row['u_outdoor']) * float(row['p_weather'])
    impact = float(row['impact_idx']) * float(row['a_foot'])
    return float(gain * dose / (half_sat + dose) - loss * impact)