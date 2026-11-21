import numpy as np

_coefficients = None

def fit(train_rows):
    global _coefficients
    X = np.array([[float(r['x_ind']) / (1.1 + float(r['x_ind'])), float(r['u_ctrl'])] for r in train_rows], dtype=float)
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    _coefficients = np.linalg.lstsq(X, y, rcond=None)[0]

def predict(row):
    x = float(row['x_ind'])
    return float(_coefficients[0] * x / (1.1 + x) + _coefficients[1] * float(row['u_ctrl']))