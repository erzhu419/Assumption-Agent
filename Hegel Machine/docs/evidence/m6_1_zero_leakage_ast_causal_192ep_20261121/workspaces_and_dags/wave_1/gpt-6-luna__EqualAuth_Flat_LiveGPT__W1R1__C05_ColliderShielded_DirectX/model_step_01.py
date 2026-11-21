import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    x = np.array([float(r['x_ind']) for r in train_rows])
    u = np.array([float(r['u_ctrl']) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    h = x / (1.1 + x)
    A = np.column_stack([h, u])
    p0 = np.linalg.lstsq(A, y, rcond=None)[0]
    result = least_squares(lambda p: A @ p - y, p0)
    return {'vmax': float(result.x[0]), 'c_ctrl': float(result.x[1])}

def predict(row):
    # Parameters are set by fit and attached to its returned model in normal use.
    raise RuntimeError('Use predict_with_model(row, model)')

def predict_with_model(row, model):
    x = float(row['x_ind'])
    u = float(row['u_ctrl'])
    return model['vmax'] * x / (1.1 + x) + model['c_ctrl'] * u
