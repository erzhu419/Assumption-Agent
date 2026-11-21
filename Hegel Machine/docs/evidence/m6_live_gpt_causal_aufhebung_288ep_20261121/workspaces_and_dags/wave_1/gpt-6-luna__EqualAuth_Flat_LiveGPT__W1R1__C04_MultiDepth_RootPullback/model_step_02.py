import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    X=np.array([[float(r['u_indoor']),float(r['a_foot']),float(r['impact_idx'])] for r in train_rows])
    y=np.array([float(r['y']) for r in train_rows])
    # Saturating indoor dose, attenuated by impact burden; fit mechanistic parameters.
    def pred(p,x):
        dose=x[:,0]/(p[0]+x[:,0])
        return p[1]+p[2]*dose-p[3]*x[:,1]*x[:,2]
    res=least_squares(lambda p: pred(p,X)-y, x0=[1.0,0.0,8.0,3.0], bounds=([0.01,-20,-50,-20],[20,20,50,20]))
    p=res.x
    def predict(row):
        x=np.array([[float(row['u_indoor']),float(row['a_foot']),float(row['impact_idx'])]])
        return float(pred(p,x)[0])
    return predict

def predict(row):
    return fit([])(row)
