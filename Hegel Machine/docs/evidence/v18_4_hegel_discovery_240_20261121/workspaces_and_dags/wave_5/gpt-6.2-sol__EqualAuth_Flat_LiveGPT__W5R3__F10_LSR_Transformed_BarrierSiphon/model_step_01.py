import math
import numpy as np
from scipy.optimize import least_squares


def fit(train_rows):
    global params
    u, s, y = np.array([(r['u_now']/0.02, r['s_grad'], r['y']) for r in train_rows], dtype=float).T
    sink = s/(1.0+s)
    root = np.sqrt(u)
    flux = u*np.exp(-1.0/(0.5+u))
    j, q, c = np.linalg.lstsq(np.column_stack((flux, -flux*sink, root)), y, rcond=None)[0]

    def residual(p):
        j, b, q, c = p
        return u*np.exp(-b/(0.5+u))*(j-q*sink)+c*root-y

    # q is the product of the flux amplitude and sink coefficient.
    params = least_squares(residual, [j, 1.0, q, c], bounds=([-np.inf, 0.0, -np.inf, -np.inf], np.inf)).x
    return params


def predict(row) -> float:
    j, b, q, c = params
    u = row['u_now']/0.02
    s = row['s_grad']
    return float(u*math.exp(-b/(0.5+u))*(j-q*s/(1.0+s))+c*math.sqrt(u))