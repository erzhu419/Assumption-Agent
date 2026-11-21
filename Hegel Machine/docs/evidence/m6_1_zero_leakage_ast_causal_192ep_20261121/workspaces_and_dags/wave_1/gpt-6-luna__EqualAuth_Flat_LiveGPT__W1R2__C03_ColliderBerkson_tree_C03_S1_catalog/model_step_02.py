import math
import numpy as np
from scipy.optimize import least_squares


def _features(row):
    u1 = float(row['u_1'])
    u2 = float(row['u_2'])
    return math.log1p(u1), u2 / (1.0 + u2)


def fit(train_rows):
    X = np.array([_features(r) for r in train_rows], dtype=float)
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    result = least_squares(lambda p: X @ p - y, x0=np.array([3.0, 4.0]))
    return result.x


def predict(row):
    # Parameters are set by fit(train_rows) and stored on the function for a
    # self-contained fit/predict interface.
    p = getattr(predict, '_parameters', None)
    if p is None:
        raise RuntimeError('Call fit(train_rows) first')
    x = np.asarray(_features(row), dtype=float)
    return float(x @ p)


def _fit_and_store(train_rows):
    p = fit(train_rows)
    predict._parameters = p
    return p

# Make fit both return coefficients and initialize predict for the normal workflow.
_original_fit = fit
def fit(train_rows):
    p = _original_fit(train_rows)
    predict._parameters = p
    return p
