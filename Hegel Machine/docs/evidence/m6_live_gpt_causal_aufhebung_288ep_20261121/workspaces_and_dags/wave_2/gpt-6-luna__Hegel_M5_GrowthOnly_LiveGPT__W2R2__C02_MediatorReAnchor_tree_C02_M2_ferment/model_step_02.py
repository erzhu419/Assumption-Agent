import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    s=np.array([float(r['s_surf']) for r in train_rows]); u=np.array([float(r['u_raw']) for r in train_rows]); y=np.array([float(r['y']) for r in train_rows])
    def pred(p,ss,uu):
        vmax,k,c=p
        return vmax*ss/(k+ss)+c*np.sqrt(uu)
    res=least_squares(lambda p: pred(p,s,u)-y, x0=[7.0,1.0,0.7], bounds=([0,1e-9,0],[100,100,100]))
    p=res.x
    def predict(row):
        return float(pred(p,float(row['s_surf']),float(row['u_raw'])))
    return predict

def predict(row):
    return float('nan')