import numpy as np
from scipy.optimize import least_squares
from itertools import combinations

_coef = None
_selected = None

def _features(row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    q = u / (1.2 + u)
    g = s / (1.0 + s)
    return np.array([q, q*g, q*g*g, g, q*q, 1.0, float(row['x_now']), float(row['mem_conv']), float(row['u_past']) - u], dtype=float)

def fit(train_rows):
    global _coef, _selected
    X = np.vstack([_features(r) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    candidates = [(0, 1)]
    for k in (1, 2):
        candidates.extend((0, 1) + c for c in combinations(range(2, X.shape[1]), k))
    best = None
    for cols in candidates:
        A = X[:, cols]
        errors = []
        for i in range(len(y)):
            mask = np.arange(len(y)) != i
            p = np.linalg.lstsq(A[mask], y[mask], rcond=None)[0]
            errors.append(float(A[i] @ p - y[i]))
        score = np.mean(np.square(errors))
        if best is None or score < best[0]:
            best = (score, cols)
    _selected = best[1]
    A = X[:, _selected]
    initial = np.linalg.lstsq(A, y, rcond=None)[0]
    _coef = least_squares(lambda p: A @ p - y, initial).x
    return _coef

def predict(row):
    return float(_features(row)[list(_selected)] @ _coef)
