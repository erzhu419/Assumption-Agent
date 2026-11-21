import math
import numpy as np
from scipy.optimize import least_squares

_model = None

def _features(row):
    return np.array([float(row['u_outdoor']) * float(row['p_weather']), float(row['u_indoor']), float(row['a_foot']), float(row['impact_idx'])])

def _value(x, p, family):
    out, indoor, area, impact = x
    g, k, c, loss = p
    if family == 0:
        envelope = math.exp(-c * area)
    elif family == 1:
        envelope = 1.0 / (1.0 + c * area)
    else:
        envelope = math.exp(-c * max(0.0, area - 1.0))
    dose = out + indoor * envelope
    return g * dose / (k + dose) - loss * impact * indoor

def fit(train_rows):
    global _model
    X = np.array([_features(r) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    def solve(indices, family):
        return least_squares(lambda p: np.array([_value(X[i], p, family) - y[i] for i in indices]), [8.0, 0.8, 0.5, 2.0], bounds=([0.01, 0.001, 0.0, 0.0], [100.0, 100.0, 10.0, 50.0]), max_nfev=3000).x
    scores = []
    for family in range(3):
        errors = []
        for i in range(len(y)):
            p = solve([j for j in range(len(y)) if j != i], family)
            errors.append((_value(X[i], p, family) - y[i]) ** 2)
        scores.append(np.mean(errors))
    family = int(np.argmin(scores))
    _model = (family, solve(list(range(len(y))), family))

def predict(row):
    family, p = _model
    return float(_value(_features(row), p, family))