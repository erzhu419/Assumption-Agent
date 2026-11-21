import math
import numpy as np
from scipy.optimize import least_squares


def fit(train_rows):
    rows = list(train_rows)
    u = np.array([float(r['u_now']) for r in rows])
    s = np.array([float(r['s_grad']) for r in rows])
    x = np.array([float(r['x_now']) for r in rows])
    y = np.array([float(r['y']) for r in rows])

    def model(p):
        vmax, rho, branch = p
        return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + branch * x

    result = least_squares(
        lambda p: model(p) - y,
        x0=np.array([6.0, 0.2, 0.0]),
        bounds=(np.array([1.0, 0.05, -2.0]), np.array([15.0, 0.95, 2.0])),
    )
    return {'vmax': float(result.x[0]), 'rho': float(result.x[1]), 'branch': float(result.x[2])}


def predict(row):
    p = fit([row])
    u = float(row['u_now'])
    s = float(row['s_grad'])
    x = float(row['x_now'])
    return p['vmax'] * (1.0 - p['rho'] * s / (1.0 + s)) * (u / (1.2 + u)) + p['branch'] * x