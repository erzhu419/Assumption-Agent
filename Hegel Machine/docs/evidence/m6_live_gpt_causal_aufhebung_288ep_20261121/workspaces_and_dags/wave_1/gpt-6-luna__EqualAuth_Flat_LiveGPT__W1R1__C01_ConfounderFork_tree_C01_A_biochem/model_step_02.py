import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    # Tree-A branch is active when z1 <= z2; equality covers the collinear pilot regime.
    def branch(r):
        return 1.0 if float(r.get('z_1', 0.0)) <= float(r.get('z_2', 0.0)) else 0.0
    def predp(p, r):
        vmax, khalf, cctrl, branch_gain = p
        x = float(r['x_ind']) / 1.5
        x2 = x*x
        return vmax*x2/(khalf+x2) - cctrl*float(r.get('u_ctrl', 0.0)) + branch_gain*branch(r)
    rows = list(train_rows)
    ys = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([predp(p,r) for r in rows]) - ys
    res = least_squares(residual, x0=[9.0, 2.0, 0.2, 0.5], bounds=([0.01,0.01,-5,-10],[100,100,5,10]))
    return res.x.tolist()

def predict(row):
    p = fit(_TRAIN_ROWS)
    x = float(row['x_ind']) / 1.5
    x2 = x*x
    active = 1.0 if float(row.get('z_1',0.0)) <= float(row.get('z_2',0.0)) else 0.0
    return float(p[0]*x2/(p[1]+x2) - p[2]*float(row.get('u_ctrl',0.0)) + p[3]*active)
