import numpy as np
from scipy.optimize import least_squares

def _pred(row, p):
    j0, beta, rho, c = p
    u = max(float(row['u_now']), 1e-9)
    s = max(float(row['s_grad']), 0.0)
    flux = j0 * u * np.exp(-beta / (0.5 + u)) * (1.0 - rho * s / (1.0 + s))
    return flux + c * np.sqrt(u)

def fit(train_rows):
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_pred(r, p) for r in train_rows]) - y
    starts = [[8, 0.5, 0.3, 0.5], [10, 1.5, 0.7, 1.0], [5, 0.1, 0.1, 1.5]]
    fits = [least_squares(residual, x0, bounds=([0.01, 0.0, 0.0, -3.0], [30.0, 10.0, 0.99, 4.0]), max_nfev=20000) for x0 in starts]
    return min(fits, key=lambda z: np.sum(z.fun**2)).x

def predict(row):
    return float(_pred(row, fit_state))