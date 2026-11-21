import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, params):
    gain, half_sat, midpoint, steepness, loss = params
    indoor = float(row['u_indoor'])
    z = steepness * (float(row['a_foot']) - midpoint)
    t = math.exp(-abs(z))
    gate = t / (1.0 + t) if z >= 0 else 1.0 / (1.0 + t)
    dose = float(row['u_outdoor']) * float(row['p_weather']) + indoor * gate
    return gain * dose / (half_sat + dose) - loss * (indoor * float(row['impact_idx'])) ** 2

def fit(train_rows):
    global _params
    rows = list(train_rows)
    targets = np.array([float(row['y']) for row in rows])
    def residuals(params):
        return np.array([_value(row, params) for row in rows]) - targets
    _params = least_squares(residuals, [8.0, 1.5, 1.5, 5.0, 0.4], bounds=(1e-8, np.inf), x_scale='jac', ftol=1e-12, xtol=1e-12, gtol=1e-12).x

def predict(row):
    return float(_value(row, _params))