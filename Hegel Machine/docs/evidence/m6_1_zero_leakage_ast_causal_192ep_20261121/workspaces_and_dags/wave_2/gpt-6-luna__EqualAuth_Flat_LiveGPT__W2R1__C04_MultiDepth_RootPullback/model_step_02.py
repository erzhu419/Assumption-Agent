import math
import numpy as np
from scipy.optimize import least_squares


def _features(row):
    indoor = float(row['u_indoor'])
    outdoor_effect = float(row['u_outdoor']) * float(row['p_weather'])
    impact = float(row['impact_idx'])
    foot = float(row['a_foot'])
    # Indoor exercise has diminishing returns; impact is a space-scaled loss.
    return np.array([
        1.0,
        indoor * indoor / (1.5 + indoor),
        outdoor_effect,
        impact * foot,
        impact,
    ], dtype=float)


def fit(train_rows):
    X = np.vstack([_features(r) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    result = least_squares(lambda b: X @ b - y, np.array([1., 2., 1., -3., 0.]))
    return result.x


def predict(row):
    # fit(train_rows) returns parameters; callers should retain them and use
    # this function's equivalent with those parameters for prediction.
    return float(_features(row) @ _MODEL_PARAMS)
