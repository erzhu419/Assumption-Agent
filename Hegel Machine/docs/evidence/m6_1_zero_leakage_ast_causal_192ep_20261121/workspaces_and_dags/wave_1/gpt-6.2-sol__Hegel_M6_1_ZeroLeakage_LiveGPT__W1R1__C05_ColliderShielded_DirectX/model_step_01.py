import math
import numpy as np
from scipy.optimize import least_squares

_coef = None


def fit(train_rows):
    global _coef
    rows = list(train_rows)
    x = np.asarray([r['x_ind'] for r in rows], dtype=float)
    u = np.asarray([r['u_ctrl'] for r in rows], dtype=float)
    y = np.asarray([r['y'] for r in rows], dtype=float)
    design = np.column_stack((x / (1.1 + x), u))
    _coef = least_squares(lambda p: design @ p - y, [1.0, 0.1], jac=lambda p: design).x


def predict(row) -> float:
    x = float(row['x_ind'])
    u = float(row['u_ctrl'])
    return math.fsum((_coef[0] * x / (1.1 + x), _coef[1] * u))
