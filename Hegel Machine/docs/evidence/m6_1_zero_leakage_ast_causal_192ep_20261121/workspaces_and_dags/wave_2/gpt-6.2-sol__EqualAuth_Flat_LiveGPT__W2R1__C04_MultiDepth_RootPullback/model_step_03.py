import math
import numpy as np
from scipy.optimize import least_squares

_params = np.array([9.205, 1.447, 1.5, 6.0, 0.473])
_fields = ('u_outdoor', 'u_indoor', 'p_weather', 'a_foot', 'impact_idx')


def fit(train_rows):
    global _params
    rows = list(train_rows)
    outdoor, indoor, weather, footprint, impact = np.array([[r[k] for k in _fields] for r in rows], dtype=float).T
    target = np.array([r['y'] for r in rows], dtype=float)

    def residual(params):
        scale, half_sat, midpoint, slope, loss = params
        gate = 1.0 / (1.0 + np.exp(np.clip(slope * (footprint - midpoint), -700, 700)))
        effort = weather * outdoor + gate * indoor
        return scale * effort / (half_sat + effort) - loss * (impact * indoor)**2 - target

    _params = least_squares(residual, [9.205, 1.447, 1.5, 6.0, 0.473], bounds=([0, 1e-8, 0, 0, 0], np.inf)).x


def predict(row) -> float:
    outdoor, indoor, weather, footprint, impact = (float(row[k]) for k in _fields)
    scale, half_sat, midpoint, slope, loss = _params
    z = max(-700.0, min(700.0, slope * (footprint - midpoint)))
    effort = weather * outdoor + indoor / (1.0 + math.exp(z))
    return float(scale * effort / (half_sat + effort) - loss * (impact * indoor)**2)
