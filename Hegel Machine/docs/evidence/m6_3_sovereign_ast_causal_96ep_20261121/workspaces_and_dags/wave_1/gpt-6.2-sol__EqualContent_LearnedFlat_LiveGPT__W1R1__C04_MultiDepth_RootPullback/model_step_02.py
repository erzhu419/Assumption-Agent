import numpy as np
from scipy.optimize import least_squares

_KEYS = ('u_outdoor', 'u_indoor', 'p_weather', 'a_foot', 'impact_idx')
_params = None


def _law(params, x):
    ceiling, half_sat, indoor_gain, midpoint, steepness, impact_cost = params
    outdoor, indoor, weather, footprint, impact = x
    gate = 1.0 / (1.0 + np.exp(np.clip(steepness * (footprint - midpoint), -60.0, 60.0)))
    effort = outdoor * weather + indoor_gain * gate * indoor
    return ceiling * effort / (half_sat + effort) - impact_cost * (indoor * impact) ** 2


def fit(train_rows):
    global _params
    rows = list(train_rows)
    x = np.array([[float(r[k]) for k in _KEYS] for r in rows]).T
    y = np.array([float(r['y']) for r in rows])
    _params = least_squares(
        lambda p: _law(p, x) - y,
        [8.24, 0.872, 0.98, 1.54, 6.0, 0.5],
        bounds=([0.01, 0.001, 0.01, 0.0, 0.01, 0.0],
                [50.0, 20.0, 10.0, 4.0, 30.0, 10.0]),
    ).x


def predict(row) -> float:
    return float(_law(_params, [float(row[k]) for k in _KEYS]))
