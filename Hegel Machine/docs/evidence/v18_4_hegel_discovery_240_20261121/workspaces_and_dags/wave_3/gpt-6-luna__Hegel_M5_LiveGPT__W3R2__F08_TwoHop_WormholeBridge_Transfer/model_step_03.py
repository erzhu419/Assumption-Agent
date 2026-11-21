import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    r = train_rows
    u = np.array([float(a['u_now']) for a in r])
    g = np.array([float(a['s_grad']) for a in r])
    y = np.array([float(a['y']) for a in r])
    def pred(p):
        h, eps, c = p
        return h*u - eps*(0.2*u)**4 + c*g*u
    res = least_squares(lambda p: pred(p)-y, x0=np.array([1.0, 0.1, 0.0]))
    return res.x

def predict(row):
    p = fit(_train_rows)
    u = float(row['u_now'])
    g = float(row['s_grad'])
    h, eps, c = p
    return float(h*u - eps*(0.2*u)**4 + c*g*u)