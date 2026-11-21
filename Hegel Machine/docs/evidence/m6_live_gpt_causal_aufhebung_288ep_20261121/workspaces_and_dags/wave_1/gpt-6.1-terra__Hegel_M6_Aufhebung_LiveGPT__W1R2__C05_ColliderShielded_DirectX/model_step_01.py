import numpy as np

def fit(train_rows):
    global coefficients
    X = np.array([[r['x_ind'] / (1.1 + r['x_ind']), r['u_ctrl']] for r in train_rows], dtype=float)
    y = np.array [r['y'] for r in train_rows]
    coefficients = np.linalg.lstsq(X, y, rcond=None)[0]

def predict(row):
    return float(coefficients[0] * row['x_ind'] / (1.1 + row['x_ind']) + coefficients[1] * row['u_ctrl'])