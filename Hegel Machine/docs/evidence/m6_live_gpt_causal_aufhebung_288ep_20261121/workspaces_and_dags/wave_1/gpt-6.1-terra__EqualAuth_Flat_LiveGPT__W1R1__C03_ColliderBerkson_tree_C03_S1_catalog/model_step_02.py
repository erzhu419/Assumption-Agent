import math
import numpy as np

_coef = None

def fit(train_rows):
    global _coef
    X = np.array([[math.log1p(float(r['u_1'])), float(r['u_2']) / (1.0 + float(r['u_2']))] for r in train_rows], dtype=float)
    y = np.array [float(r['y']) for r in train_rows]
    _coef = np.linalg.lstsq(X, np.asarray(y), rcond=None)[0]

def predict(row) -> float:
    u1, u2 = float(row['u_1']), float(row['u_2'])
    return float(_coef[0] * math.log1p(u1) + _coef[1] * u2 / (1.0 + u2))