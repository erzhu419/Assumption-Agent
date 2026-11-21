import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    def features(r):
        outdoor = float(r['u_outdoor']) * float(r['p_weather'])
        indoor = float(r['u_indoor']) / (1.0 + math.exp(6.0 * (float(r['a_foot']) - 1.5)))
        penalty_base = (float(r['u_indoor']) * float(r['impact_idx'])) ** 2
        return outdoor + indoor, penalty_base
    def residual(params):
        g, k, c = params
        return np.array([g * dose / (k + dose) - c * penalty - float(r['y'])
                         for r, (dose, penalty) in zip(rows, map(features, rows))])
    result = least_squares(residual, x0=np.array([8.0, 1.0, 1.0]), bounds=([0.0, 1e-8, 0.0], [100.0, 100.0, 100.0]))
    return tuple(float(v) for v in result.x)

def predict(row):
    g, k, c = fit(_fit_rows)
    dose = float(row['u_outdoor']) * float(row['p_weather']) + float(row['u_indoor']) / (1.0 + math.exp(6.0 * (float(row['a_foot']) - 1.5)))
    loss = c * (float(row['u_indoor']) * float(row['impact_idx'])) ** 2
    return g * dose / (k + dose) - loss
