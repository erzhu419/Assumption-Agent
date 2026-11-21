import math
import numpy as np
from scipy.optimize import least_squares

_FIELDS = ('u_outdoor', 'u_indoor', 'p_weather', 'a_foot', 'impact_idx')
_params = None


def fit(train_rows):
    global _params
    rows = list(train_rows)
    outdoor, indoor, weather, footprint, impact = np.array(
        [[r[key] for key in _FIELDS] for r in rows], dtype=float
    ).T
    observed = np.array([r['y'] for r in rows], dtype=float)

    def residuals(params):
        gain, half_dose, steepness, midpoint, loss = params
        space = np.exp(-np.logaddexp(0.0, steepness * (footprint - midpoint)))
        dose = outdoor * weather + indoor * space
        return gain * dose / (half_dose + dose) - loss * (indoor * impact)**2 - observed

    _params = least_squares(
        residuals, [8.352, 1.528, 6.0, 1.5, 0.43],
        bounds=([0.0, 1e-8, 0.0, 0.0, 0.0], np.inf),
    ).x


def predict(row) -> float:
    gain, half_dose, steepness, midpoint, loss = _params
    outdoor, indoor, weather, footprint, impact = (float(row[key]) for key in _FIELDS)
    z = steepness * (footprint - midpoint)
    space = math.exp(-float(np.logaddexp(0.0, z)))
    dose = outdoor * weather + indoor * space
    return float(gain * dose / (half_dose + dose) - loss * (indoor * impact)**2)
