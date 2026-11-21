import numpy as np

_coef = None

def fit(train_rows):
    global _coef
    X = np.array([[r['x_ind'] / (1.1 + r['x_ind']), r['u_ctrl']] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    _coef = np.linalg.lstsq(X, y, rcond=None)[0]

def predict(row) -> float:
    return float(_coef[0] * row['x_ind'] / (1.1 + row['x_ind']) + _coef[1] * row['u_ctrl'])