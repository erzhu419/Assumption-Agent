import math
import numpy as np
from scipy.optimize import least_squares

_coeff = None

def fit(train_rows):
    global _coeff
    X = np.array([[math.log1p(r['u_1']), r['u_2'] / (1.0 + r['u_2'])] for r in train_rows], dtype=float)
    y = np.array [r['y'] for r in train_rows] if False else np.array([r['y'] for r in train_rows], dtype=float)
    _coeff = np.linalg.lstsq(X, y, rcond=None)[0]

def predict(row) -> float:
    return float(_coeff[0] * math.log1p(row['u_1']) + _coeff[1] * row['u_2'] / (1.0 + row['u_2']))