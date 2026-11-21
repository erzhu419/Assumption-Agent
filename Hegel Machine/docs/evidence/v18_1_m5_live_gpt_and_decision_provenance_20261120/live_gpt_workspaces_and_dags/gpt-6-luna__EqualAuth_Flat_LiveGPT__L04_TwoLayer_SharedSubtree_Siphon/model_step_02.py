import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    rows = train_rows
    u = np.array([r['u_now'] for r in rows], dtype=float)
    s = np.array([r['s_grad'] for r in rows], dtype=float)
    x = np.array([r['x_now'] for r in rows], dtype=float)
    m = np.array([r['mem_conv'] for r in rows], dtype=float)
    y = np.array([r['y'] for r in rows], dtype=float)
    def pred(p, u, s, x, m):
        vmax, rho, bx, bm = p
        return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + bx*x + bm*m
    res = least_squares(lambda p: pred(p,u,s,x,m)-y, x0=[5.5,0.7,0.0,0.0], bounds=([0.0,0.0,-10.0,-10.0],[20.0,1.0,10.0,10.0]))
    return res.x

def predict(row):
    p = fit(_train_rows)
    u=float(row['u_now']); s=float(row['s_grad']); x=float(row['x_now']); m=float(row['mem_conv'])
    vmax,rho,bx,bm=p
    return float(vmax*(1.0-rho*s/(1.0+s))*(u/(1.2+u))+bx*x+bm*m)
