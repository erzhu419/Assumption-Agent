import numpy as np
from scipy.optimize import least_squares
import math

def fit(train_rows):
    rows = list(train_rows)
    u = np.array([float(r['u_now']) for r in rows])
    s = np.array([float(r['s_grad']) for r in rows])
    y = np.array([float(r['y']) for r in rows])
    def pred(p, r):
        vmax, kh, alpha, leak = p
        un = max(0.0, float(r['u_now']))
        sg = float(r['s_grad'])
        hill = un*un/(kh + un*un)
        return vmax*hill/(1.0 + alpha*sg) - leak*un
    def residual(p):
        return np.array([pred(p, r) for r in rows]) - y
    best = None
    for init in ([5, 1, .5, .1], [10, 4, .2, .3], [3, .3, 1, .05]):
        res = least_squares(residual, init, bounds=([0, 1e-6, 0, 0], [100, 100, 20, 20]), max_nfev=20000)
        if best is None or np.dot(res.fun, res.fun) < np.dot(best.fun, best.fun):
            best = res
    return best.x

def predict(row):
    p = fit([{'u_now':.5,'s_grad':.99,'y':1.1742852},{'u_now':1.2,'s_grad':1.008,'y':2.0127887},{'u_now':1.9,'s_grad':.995,'y':2.471164},{'u_now':2.7,'s_grad':1.005,'y':2.7510365},{'u_now':3.5,'s_grad':.992,'y':2.9449842},{'u_now':2.2,'s_grad':.18,'y':4.2995057},{'u_now':2.9,'s_grad':2.35,'y':1.7285628},{'u_now':.8,'s_grad':.24,'y':2.5405243},{'u_now':1.7,'s_grad':2.1,'y':1.5735144},{'u_now':3.1,'s_grad':.38,'y':4.084573}])
    vmax, kh, alpha, leak = p
    un = max(0.0, float(row['u_now']))
    sg = float(row['s_grad'])
    return float(vmax*(un*un/(kh+un*un))/(1.0+alpha*sg) - leak*un)
