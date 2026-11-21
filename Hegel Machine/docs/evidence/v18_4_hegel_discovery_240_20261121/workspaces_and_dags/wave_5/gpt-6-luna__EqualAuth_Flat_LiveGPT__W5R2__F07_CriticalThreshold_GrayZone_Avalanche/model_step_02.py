import math
import numpy as np
from scipy.optimize import least_squares


def _value(row, p):
    vmax, rho, c_loc, k_aval, s_crit = p
    u = max(1e-8, float(row['u_now']))
    s = float(row['s_grad'])
    return (vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
            + c_loc * math.sqrt(u)
            - k_aval * max(0.0, s - s_crit) * u)


def fit(train_rows):
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows])
    def residual(p):
        return np.asarray([_value(r, p) for r in rows]) - y
    starts = [np.array([4.0, 0.4, 0.4, 1.0, 2.6]),
              np.array([3.0, 0.3, 0.5, 1.5, 2.5]),
              np.array([5.0, 0.5, 0.2, 0.8, 2.7])]
    fits = [least_squares(residual, p0,
                          bounds=([0.0, 0.0, 0.0, 0.0, 0.0],
                                  [100.0, 5.0, 10.0, 20.0, 10.0]),
                          max_nfev=20000) for p0 in starts]
    return min(fits, key=lambda z: float(np.dot(z.fun, z.fun))).x


def predict(row):
    p = fit(_TRAIN_ROWS)
    return float(_value(row, p))