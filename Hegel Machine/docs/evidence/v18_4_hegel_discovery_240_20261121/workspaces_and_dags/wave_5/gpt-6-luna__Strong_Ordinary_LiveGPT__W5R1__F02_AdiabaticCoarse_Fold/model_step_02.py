import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    res = least_squares(lambda p: p[0] * u - y, x0=np.array([2.2]))
    return {'gain': float(res.x[0])}

def predict(row):
    # Fit is called by the harness; retain its fitted gain for row prediction.
    global _model_params
    return float(_model_params['gain'] * row['u_now'])

_model_params = {'gain': 2.24}
_original_fit = fit
def fit(train_rows):
    global _model_params
    _model_params = _original_fit(train_rows)
    return _model_params
