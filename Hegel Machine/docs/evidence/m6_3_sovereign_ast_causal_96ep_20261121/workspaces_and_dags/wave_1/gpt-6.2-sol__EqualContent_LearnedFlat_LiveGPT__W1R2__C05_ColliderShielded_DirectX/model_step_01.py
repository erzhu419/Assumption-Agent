import math
import numpy as np
from scipy.optimize import least_squares


def fit(train_rows):
    rows = list(train_rows)
    x = np.asarray([float(row['x_ind']) for row in rows])
    u = np.asarray([float(row['u_ctrl']) for row in rows])
    y = np.asarray([float(row['y']) for row in rows])
    design = np.column_stack((x / (1.1 + x), u))
    vmax, c_ctrl = least_squares(lambda p: design @ p - y, [1.0, 0.0], jac=lambda p: design).x

    def predict(row) -> float:
        x = float(row['x_ind'])
        u = float(row['u_ctrl'])
        return math.fsum((vmax * (x / (1.1 + x)), c_ctrl * u))

    return predict