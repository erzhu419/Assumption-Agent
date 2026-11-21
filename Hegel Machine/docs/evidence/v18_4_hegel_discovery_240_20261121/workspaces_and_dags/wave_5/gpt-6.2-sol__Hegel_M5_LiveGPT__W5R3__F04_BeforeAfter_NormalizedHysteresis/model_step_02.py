import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _response(row, params):
    gain, kick, rate = params
    u_now = float(row['u_now'])
    step = u_now - float(row['u_past'])
    equilibrium = gain * u_now
    decay = math.exp(-rate * float(row['H']))
    return equilibrium + (float(row['x_now']) - equilibrium - kick * (step / 0.02)) * decay

def fit(train_rows):
    global _params
    rows = list(train_rows)
    targets = np.asarray([float(row['y']) for row in rows])
    def residuals(params):
        return np.asarray([_response(row, params) for row in rows]) - targets
    _params = least_squares(residuals, [15.0, 2.0, 1.5], bounds=([-np.inf, -np.inf, 0.0], [np.inf, np.inf, np.inf]), ftol=1e-12, xtol=1e-12, gtol=1e-12).x

def predict(row) -> float:
    return float(_response(row, _params))