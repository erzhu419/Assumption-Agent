import numpy as np
from scipy.optimize import least_squares

def _features(row):
    p = float(row['p_weather'])
    uo = float(row['u_outdoor'])
    ui = float(row['u_indoor'])
    area = float(row['a_foot'])
    impact = float(row['impact_idx'])
    # Indoor activity is attenuated by space/footprint; rainy outdoor activity
    # contributes only in proportion to usable weather.
    dose = max(0.0, uo * p + ui / max(area, 1e-6))
    loss = impact * area
    return dose, loss

def fit(train_rows):
    X = np.array([_features(r) for r in train_rows], dtype=float)
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    def pred(theta, x):
        gain, half, exponent, loss_gain, offset = theta
        d, loss = x[:, 0], x[:, 1]
        return offset + gain * (d ** exponent) / (half ** exponent + d ** exponent) - loss_gain * loss
    init = np.array([8.0, 1.0, 1.0, 3.0, 0.0])
    lo = [0.0, 1e-4, 0.2, 0.0, -10.0]
    hi = [30.0, 20.0, 3.0, 20.0, 10.0]
    result = least_squares(lambda t: pred(t, X) - y, init, bounds=(lo, hi), max_nfev=10000)
    return result.x

def predict(row):
    # Parameters are populated by fit; retain them in module-level state.
    d, loss = _features(row)
    gain, half, exponent, loss_gain, offset = _PARAMS
    return float(offset + gain * (d ** exponent) / (half ** exponent + d ** exponent) - loss_gain * loss)

_PARAMS = np.array([8.0, 1.0, 1.0, 3.0, 0.0])
_fit_original = fit
def fit(train_rows):
    global _PARAMS
    _PARAMS = _fit_original(train_rows)
    return _PARAMS