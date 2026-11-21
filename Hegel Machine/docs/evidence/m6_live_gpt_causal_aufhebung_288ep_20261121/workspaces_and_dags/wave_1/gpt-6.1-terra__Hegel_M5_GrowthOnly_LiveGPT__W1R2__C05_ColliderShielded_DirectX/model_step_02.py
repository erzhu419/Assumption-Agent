import numpy as np

_coefficients = None

def fit(train_rows):
    global _coefficients
    X = np.array([[r['x_ind'] / (1.1 + r['x_ind']), r['u_ctrl']] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    _coefficients = np.linalg.lstsq(X, y, rcond=None)[0]

def predict(row) -> float:
    x = row['x_ind']
    return float(_coefficients[0] * x / (1.1 + x) + _coefficients[1] * row['u_ctrl'])
