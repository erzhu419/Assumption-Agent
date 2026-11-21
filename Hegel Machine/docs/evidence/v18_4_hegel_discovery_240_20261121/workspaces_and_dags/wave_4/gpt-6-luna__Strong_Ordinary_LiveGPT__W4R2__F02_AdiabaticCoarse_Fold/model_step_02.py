import math
import numpy as np
from scipy.optimize import least_squares


def fit(train_rows):
    u = np.asarray([r['u_now'] for r in train_rows], dtype=float)
    y = np.asarray([r['y'] for r in train_rows], dtype=float)
    result = least_squares(lambda p: p[0] * u - y, x0=np.array([2.0]))
    return float(result.x[0])


def predict(row):
    return float(fit_params[0] * row['u_now'])
