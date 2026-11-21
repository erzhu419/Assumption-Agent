import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    X=np.array([[r['u_now'], (0.2*r['u_now'])**4, r['s_grad'], 1.0] for r in train_rows],float)
    y=np.array([r['y'] for r in train_rows],float)
    # y = a*u - b*(0.2u)^4 + c*s_grad + d
    A=X.copy(); A[:,1]*=-1
    p=np.linalg.lstsq(A,y,rcond=None)[0]
    def residual(q):
        return A@q-y
    q=least_squares(residual,p).x
    return q

def predict(row):
    q=fit(_train_rows)
    u=row['u_now']
    return float(q[0]*u-q[1]*(0.2*u)**4+q[2]*row['s_grad']+q[3])
