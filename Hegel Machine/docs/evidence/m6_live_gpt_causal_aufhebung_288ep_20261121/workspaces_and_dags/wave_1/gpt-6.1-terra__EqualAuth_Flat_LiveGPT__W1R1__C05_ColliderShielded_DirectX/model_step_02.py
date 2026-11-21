import numpy as np

_coef = None

def fit(train_rows):
    global _coef
    X = np.array([[float(r['x_ind']) / (1.1 + float(r['x_ind'])), float(r['u_ctrl'])] for r in train_rows], dtype=float)
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    _coef = np.linalg.lstsq(X, y, rcond=None)[0]

def predict(row):
    x = float(row['x_ind'])
    return float(_coef[0] * x / (1.1 + x) + _coef[1] * float(row['u_ctrl']))