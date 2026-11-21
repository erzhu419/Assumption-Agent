import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    H = np.array([r['H'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def pred(p, uu, hh):
        E, eta = p
        z = np.clip(E * hh / eta, 0.0, 700.0)
        return (uu / E) * (-np.expm1(-z))
    result = least_squares(lambda p: pred(p, u, H) - y,
                           x0=np.array([0.5, 0.5]),
                           bounds=([1e-8, 1e-8], [1e6, 1e6]),
                           max_nfev=20000)
    E, eta = result.x
    def predict(row):
        z = np.clip(E * float(row['H']) / eta, 0.0, 700.0)
        return float((float(row['u_now']) / E) * (-np.expm1(-z)))
    return predict

def predict(row):
    return fit([])(row)
