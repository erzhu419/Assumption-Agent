import numpy as np

_vmax = None
_c_ctrl = None

def fit(train_rows):
    global _vmax, _c_ctrl
    X = np.array([[float(r['x_ind']) / (1.1 + float(r['x_ind'])), float(r['u_ctrl'])] for r in train_rows], dtype=float)
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    _vmax, _c_ctrl = np.linalg.lstsq(X, y, rcond=None)[0]

def predict(row):
    x = float(row['x_ind'])
    return float(_vmax * x / (1.1 + x) + _c_ctrl * float(row['u_ctrl']))