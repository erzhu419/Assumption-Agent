import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    result = least_squares(lambda p: p[0] * u - y, x0=np.array([2.0]))
    return float(result.x[0])

def predict(row):
    return predict.slope * float(row['u_now'])

# fit() stores the fitted coefficient for predict().
_original_fit = fit
def fit(train_rows):
    slope = _original_fit(train_rows)
    predict.slope = slope
    return slope
