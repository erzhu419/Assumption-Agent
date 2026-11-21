import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    x = np.array([[math.log1p(float(r['u_1'])), float(r['u_2'])/(1.0+float(r['u_2']))] for r in train_rows], dtype=float)
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    res = least_squares(lambda a: x @ a - y, np.zeros(2))
    a1, a2 = res.x
    def predict(row):
        return float(a1*math.log1p(float(row['u_1'])) + a2*float(row['u_2'])/(1.0+float(row['u_2'])))
    return predict

def predict(row):
    raise RuntimeError('Call fit(train_rows) and use its returned predictor.')