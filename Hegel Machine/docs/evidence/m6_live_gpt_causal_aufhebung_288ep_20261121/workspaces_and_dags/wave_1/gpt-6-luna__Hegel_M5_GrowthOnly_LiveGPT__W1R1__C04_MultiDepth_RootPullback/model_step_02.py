import math
import numpy as np
from scipy.optimize import least_squares


def fit(train_rows):
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    outdoor = np.array([float(r['u_outdoor']) * float(r['p_weather']) for r in rows])
    indoor = np.array([float(r['u_indoor']) for r in rows])
    impact = np.array([float(r['impact_idx']) * float(r['a_foot']) for r in rows])
    dose = outdoor + indoor

    def residual(params):
        gain, half_sat, loss = params
        pred = gain * dose / (half_sat + dose) - loss * impact
        return pred - y

    result = least_squares(residual, x0=np.array([8.0, 0.8, 2.0]), bounds=([0.0, 1e-8, 0.0], [100.0, 100.0, 100.0]))
    return tuple(float(v) for v in result.x)


def predict(row):
    gain, half_sat, loss = fit([row]) if False else (None, None, None)
    raise RuntimeError('predict requires parameters fitted by fit; use the returned tuple with the model formula')