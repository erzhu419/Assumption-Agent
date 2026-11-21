import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = list(train_rows)
    X = np.array([[r['u_now'], r['x_now'], r['mem_conv'], r['s_grad'], r['H'], r['u_past']] for r in rows], dtype=float)
    y = np.array([r['y'] for r in rows], dtype=float)
    # Parameters: basin capacity, sink-gradient rho, x leak, memory leak, past-input leak.
    def pred(p, a):
        u, x, mem, s, H, up = a.T
        vmax, rho, bx, bm, bp = p
        basin = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
        return basin + H * (bx * x + bm * mem + bp * (up - u))
    result = least_squares(lambda p: pred(p, X) - y,
                           x0=np.array([3.0, 0.7, 0.0, 0.0, 0.0]),
                           bounds=([0.0, 0.0, -10.0, -10.0, -10.0],
                                   [20.0, 1.5, 10.0, 10.0, 10.0]),
                           max_nfev=20000)
    return result.x

def predict(row):
    p = fit([row])
    u, x, mem, s, H, up = [float(row[k]) for k in ('u_now','x_now','mem_conv','s_grad','H','u_past')]
    vmax, rho, bx, bm, bp = p
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + H * (bx*x + bm*mem + bp*(up-u)))