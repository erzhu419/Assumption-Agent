import math
import numpy as np
from scipy.optimize import least_squares

_params = np.array([8.255, 0.874, 6.0, 1.5, 0.47])

def _evaluate(row, params):
    gain, half_sat, steepness, midpoint, impact_cost = params
    indoor = float(row['u_indoor'])
    x = steepness * (float(row['a_foot']) - midpoint)
    if x >= 0:
        exp_neg = math.exp(-x)
        gate = exp_neg / (1.0 + exp_neg)
    else:
        gate = 1.0 / (1.0 + math.exp(x))
    effort = float(row['p_weather']) * float(row['u_outdoor']) + indoor * gate
    impact_dose = indoor * float(row['impact_idx'])
    return gain * effort / (half_sat + effort) - impact_cost * impact_dose**2

def fit(train_rows):
    global _params
    rows = list(train_rows)
    target = np.array([float(row['y']) for row in rows])
    def residuals(params):
        return np.array([_evaluate(row, params) for row in rows]) - target
    _params = least_squares(residuals, [8.255, 0.874, 6.0, 1.5, 0.47], bounds=([0.0, 1e-8, 0.0, -np.inf, 0.0], np.inf)).x

def predict(row) -> float:
    return float(_evaluate(row, _params))