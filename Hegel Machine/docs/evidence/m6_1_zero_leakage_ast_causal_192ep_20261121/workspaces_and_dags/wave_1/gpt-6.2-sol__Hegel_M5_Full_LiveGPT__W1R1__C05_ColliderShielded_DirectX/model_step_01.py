import math
import numpy as np
from scipy.optimize import least_squares

_coef = None

def fit(train_rows):
    global _coef
    data = np.array([(r['x_ind'], r['u_ctrl'], r['y']) for r in train_rows], dtype=float)
    x, u, y = data.T
    design = np.column_stack((x / (1.1 + x), u))
    _coef = least_squares(lambda p: design @ p - y, [1.0, 0.0], jac=lambda p: design).x

def predict(row) -> float:
    x = float(row['x_ind'])
    u = float(row['u_ctrl'])
    return math.fsum((float(_coef[0]) * x / (1.1 + x), float(_coef[1]) * u))
