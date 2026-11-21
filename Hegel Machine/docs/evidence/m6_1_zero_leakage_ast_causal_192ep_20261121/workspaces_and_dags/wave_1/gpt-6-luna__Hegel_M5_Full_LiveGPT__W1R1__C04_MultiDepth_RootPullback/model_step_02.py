import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    def dose(r):
        return float(r['u_outdoor']) * float(r['p_weather']) + float(r['u_indoor'])
    def pred_params(p, r):
        g, k, loss = p
        e = dose(r)
        return g * e / (k + e) - loss * float(r['impact_idx']) * float(r['a_foot'])
    y = np.array([float(r['y']) for r in rows])
    x0 = np.array([8.0, 0.8, 3.0])
    result = least_squares(lambda p: np.array([pred_params(p, r) for r in rows]) - y,
                           x0, bounds=([0.0, 1e-8, 0.0], [100.0, 100.0, 100.0]))
    g, k, loss = result.x
    return {'g': float(g), 'k': float(k), 'loss': float(loss)}

def predict(row):
    p = fit([row]) if False else None
    params = _MODEL_PARAMS
    e = float(row['u_outdoor']) * float(row['p_weather']) + float(row['u_indoor'])
    return float(params['g'] * e / (params['k'] + e) - params['loss'] * float(row['impact_idx']) * float(row['a_foot']))
