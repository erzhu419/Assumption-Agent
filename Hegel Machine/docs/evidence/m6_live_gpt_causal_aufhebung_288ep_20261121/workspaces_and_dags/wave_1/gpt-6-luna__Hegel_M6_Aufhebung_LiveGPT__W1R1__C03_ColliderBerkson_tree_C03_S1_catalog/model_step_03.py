import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    x1 = np.array([math.log1p(float(r['u_1'])) for r in train_rows])
    x2 = np.array([float(r['u_2']) / (1.0 + float(r['u_2'])) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return p[0] * x1 + p[1] * x2 - y
    p = least_squares(residual, np.array([1.0, 1.0])).x
    return {'coefficients': p.tolist()}

def predict(row):
    c = _model['coefficients']
    return float(c[0] * math.log1p(float(row['u_1'])) + c[1] * float(row['u_2']) / (1.0 + float(row['u_2'])))
