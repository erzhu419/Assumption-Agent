import math
import numpy as np
from scipy.optimize import least_squares

_params = None


def _features(row):
    return (float(row['u_outdoor']) * float(row['p_weather']),
            float(row['u_indoor']), float(row['a_foot']),
            float(row['impact_idx']))


def fit(train_rows):
    global _params
    rows = list(train_rows)
    outdoor, indoor, area, impact = np.array([_features(r) for r in rows]).T
    y = np.array([float(r['y']) for r in rows])
    exposure_sq = (indoor * impact) ** 2

    def residual(params):
        gain, half_dose, midpoint, steepness, loss = params
        z = steepness * (area - midpoint)
        gate = np.exp(-np.logaddexp(0.0, z))
        dose = outdoor + indoor * gate
        return gain * dose / (half_dose + dose) - loss * exposure_sq - y

    result = least_squares(residual, [8.255, 0.874, 1.5, 6.0, 0.47],
                           bounds=([0.0, 1e-8, -np.inf, 0.0, 0.0], np.inf))
    _params = tuple(map(float, result.x))
    return _params


def predict(row) -> float:
    outdoor, indoor, area, impact = _features(row)
    gain, half_dose, midpoint, steepness, loss = _params
    z = steepness * (area - midpoint)
    t = math.exp(-abs(z))
    gate = t / (1.0 + t) if z >= 0.0 else 1.0 / (1.0 + t)
    dose = outdoor + indoor * gate
    return float(gain * dose / (half_dose + dose) - loss * (indoor * impact) ** 2)
