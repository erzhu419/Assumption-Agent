import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    x1=np.array([float(r['u_1']) for r in train_rows]); x2=np.array([float(r['u_2']) for r in train_rows]); y=np.array([float(r['y']) for r in train_rows])
    def pred(p,a,b): return p[0]*np.log1p(a)+p[1]*b/(p[2]+b)
    def res(p): return pred(p,x1,x2)-y
    p=least_squares(res,[2.0,4.0,1.0],bounds=([-20,-20,1e-6],[20,20,100])).x
    def predict(row):
        a=float(row['u_1']); b=float(row['u_2'])
        return float(pred(p,a,b))
    return predict

def predict(row):
    raise RuntimeError('Call fit(train_rows) first and use its returned predictor.')