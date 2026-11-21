import math
import numpy as np
from scipy.optimize import least_squares

_params = np.array([8.352, 1.528, 6.0, 0.43])


def fit(train_rows):
    global _params
    cols = ('u_outdoor', 'u_indoor', 'p_weather', 'a_foot', 'impact_idx', 'y')
    outdoor, indoor, weather, footprint, impact, y = np.array(
        [[float(row[key]) for key in cols] for row in train_rows], dtype=float
    ).T
    outdoor_eff = outdoor * weather
    penalty = (indoor * impact) ** 2

    def residual(params):
        scale, half_sat, steepness, loss = params
        gate = np.exp(-np.logaddexp(0.0, steepness * (footprint - 1.5)))
        eff = outdoor_eff + indoor * gate
        return scale * eff / (half_sat + eff) - loss * penalty - y

    _params = least_squares(residual, [8.35, 1.53, 6.0, 0.43],
                            bounds=([0.0, 1e-8, 0.0, 0.0], np.inf)).x


def predict(row) -> float:
    scale, half_sat, steepness, loss = _params
    indoor = float(row['u_indoor'])
    z = steepness * (float(row['a_foot']) - 1.5)
    gate = math.exp(-float(np.logaddexp(0.0, z)))
    eff = float(row['u_outdoor']) * float(row['p_weather']) + indoor * gate
    return float(scale * eff / (half_sat + eff) - loss * (indoor * float(row['impact_idx'])) ** 2)
