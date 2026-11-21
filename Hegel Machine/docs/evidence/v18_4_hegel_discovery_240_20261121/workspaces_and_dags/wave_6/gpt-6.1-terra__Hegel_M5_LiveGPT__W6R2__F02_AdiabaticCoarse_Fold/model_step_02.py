import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    x = np.array([r['x_now'] for r in train_rows], dtype=float)
    h = np.array([r['H'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)

    def residual(p):
        a, b, c, k = p
        return a*u + np.exp(-k*h)*(b*x + c*u) - y

    candidates = [least_squares(residual, [2.5, 0.5, -1.0, k],
                                bounds=([-20, -100, -100, 0.001],
                                        [20, 100, 100, 3.0]),
                                max_nfev=3000)
                  for k in (0.2, 0.7, 1.2)]
    _params = min(candidates, key=lambda result: np.sum(result.fun**2)).x

def predict(row):
    a, b, c, k = _params
    return float(a*row['u_now'] + np.exp(-k*row['H'])*(b*row['x_now'] + c*row['u_now']))