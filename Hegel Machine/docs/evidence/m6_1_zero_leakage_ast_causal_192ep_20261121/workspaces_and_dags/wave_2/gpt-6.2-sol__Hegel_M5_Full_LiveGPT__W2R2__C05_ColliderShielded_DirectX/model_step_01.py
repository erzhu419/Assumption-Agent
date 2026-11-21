import math
import numpy as np
from scipy.optimize import least_squares

_params = None


def _actuator(x):
    return x / math.fsum((1.1, x))


def fit(train_rows):
    global _params
    data = np.asarray([(_actuator(float(r['x_ind'])), float(r['u_ctrl']), float(r['y'])) for r in train_rows], dtype=float)
    design = data[:, :2]
    target = data[:, 2]
    result = least_squares(lambda p: design @ p - target, [1.0, 0.0], jac=lambda p: design)
    _params = result.x
    return _params


def predict(row):
    vmax, c_ctrl = _params
    return float(vmax * _actuator(float(row['x_ind'])) + c_ctrl * float(row['u_ctrl']))