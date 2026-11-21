import math
import numpy as np
from scipy.optimize import least_squares

_coeff = None

def fit(train_rows):
    global _coeff
    X = np.array([[math.log1p(float(r['u_1'])), float(r['u_2']) / (1.0 + float(r['u_2']))] for r in train_rows], dtype=float)
    y = np.array [float(r['y']) for r in train_rows]
    _coeff = np.linalg.lstsq(X, y, rcond=None)[0]

def predict(row) -> float:
    return float(_coeff[0] * math.log1p(float(row['u_1'])) + _coeff[1] * float(row['u_2']) / (1.0 + float(row['u_2'])))