import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    def residual(p):
        a, b, k = p
        pred = [r['x_now'] + (r['u_past'] - r['u_now']) * (a + b*r['x_now']) * np.exp(-k*r['H']) for r in rows]
        return np.asarray(pred) - np.asarray([r['y'] for r in rows])
    sol = least_squares(residual, x0=np.array([400.0, 0.0, 0.5]), max_nfev=20000)
    return sol.x

def predict(row):
    a, b, k = fit(_fit_rows)
    return float(row['x_now'] + (row['u_past'] - row['u_now']) * (a + b*row['x_now']) * np.exp(-k*row['H']))

_fit_rows = []

# fit is normally called once on the supplied training rows; retain the fitted data for predict.
_fit_impl = fit
def fit(train_rows):
    global _fit_rows
    _fit_rows = list(train_rows)
    return _fit_impl(_fit_rows)
