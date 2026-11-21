import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    a = np.array([[r['x_ind'] / (1.1 + r['x_ind']), r['u_ctrl']] for r in train_rows], dtype=float)
    y = np.array [r['y'] for r in train_rows]
    _params = np.linalg.lstsq(a, y, rcond=None)[0]

def predict(row) -> float:
    return float(_params[0] * row['x_ind'] / (1.1 + row['x_ind']) + _params[1] * row['u_ctrl'])