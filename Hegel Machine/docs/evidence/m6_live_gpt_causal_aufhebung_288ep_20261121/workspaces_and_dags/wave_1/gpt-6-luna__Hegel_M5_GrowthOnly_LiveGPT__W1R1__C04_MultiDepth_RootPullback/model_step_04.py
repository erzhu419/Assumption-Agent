import numpy as np
from scipy.optimize import least_squares

_params = None

def _dose(row):
    return max(0.0, float(row['u_indoor']) + float(row['u_outdoor']) * float(row['p_weather']))

def predict(row):
    if _params is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    gain, half_sat, loss = _params
    e = _dose(row)
    return float(gain * e / (half_sat + e) - loss * float(row['impact_idx']) * float(row['a_foot']))

def fit(train_rows):
    global _params
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        gain, half_sat, loss = p
        pred = []
        for r in train_rows:
            e = _dose(r)
            pred.append(gain * e / (half_sat + e) - loss * float(r['impact_idx']) * float(r['a_foot']))
        return np.asarray(pred) - y
    result = least_squares(residual, x0=[8.0, 1.5, 4.0], bounds=([0.0, 1e-6, 0.0], [50.0, 50.0, 50.0]))
    _params = tuple(float(v) for v in result.x)
    return _params