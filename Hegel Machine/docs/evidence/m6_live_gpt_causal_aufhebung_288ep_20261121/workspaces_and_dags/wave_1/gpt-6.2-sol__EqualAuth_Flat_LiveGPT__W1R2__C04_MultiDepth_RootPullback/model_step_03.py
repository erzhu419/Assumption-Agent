import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    x = np.array([[r['u_outdoor'] * r['p_weather'], r['u_indoor'], r['a_foot'], r['impact_idx']] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    outdoor, indoor, footprint, impact = x.T

    def residual(p):
        gain, half_dose, steepness, loss = p
        envelope = np.exp(-np.logaddexp(0.0, steepness * (footprint - 1.5)))
        dose = outdoor + indoor * envelope
        return gain * dose / (half_dose + dose) - loss * (indoor * impact)**2 - y

    _params = least_squares(residual, [8.0, 1.5, 6.0, 0.4], bounds=([0.0, 1e-8, 0.0, 0.0], np.inf), gtol=1e-12).x

def predict(row) -> float:
    gain, half_dose, steepness, loss = _params
    envelope = math.exp(-float(np.logaddexp(0.0, steepness * (row['a_foot'] - 1.5))))
    indoor = row['u_indoor']
    dose = row['u_outdoor'] * row['p_weather'] + indoor * envelope
    return float(gain * dose / (half_dose + dose) - loss * (indoor * row['impact_idx'])**2)
