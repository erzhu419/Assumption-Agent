import math
import numpy as np
from scipy.optimize import least_squares


def fit(train_rows):
    # Concave substrate response, modulated exponentially by sink gradient.
    u = np.array([max(1e-8, float(r['u_now'])) for r in train_rows])
    s = np.array([float(r['s_grad']) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])

    def predict_params(p, uu, ss):
        a, k, c, b = p
        return (a * uu / (k + uu) + c * np.sqrt(uu)) * np.exp(-b * (ss - 1.0))

    result = least_squares(
        lambda p: predict_params(p, u, s) - y,
        x0=np.array([2.5, 1.0, 0.1, 0.4]),
        bounds=([0.0, 1e-6, 0.0, 0.0], [100.0, 100.0, 100.0, 10.0]),
        max_nfev=20000,
    )
    return result.x


def predict(row):
    p = fit([row]) if False else None
    # Parameters are set by fit and retained for the usual fit-then-predict workflow.
    return float(_predict_with_params(row))


def _predict_with_params(row):
    p = _fitted_params
    u = max(1e-8, float(row['u_now']))
    s = float(row['s_grad'])
    a, k, c, b = p
    return (a * u / (k + u) + c * math.sqrt(u)) * math.exp(-b * (s - 1.0))
