import pickle, sys
sys.path.insert(0, '/holdout_task')
import numpy as np
from scipy.optimize import least_squares
from upstream_transducer import _fit_forward, _inverse
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    study = pickle.load(f)[0]['study']
s = study['sensor_calibration_sweeps']
x, y = s['ref_rosette'], s['raw_rosette']
j = np.arange(4); nxt = (j+1)%4; prev = (j-1)%4

def forward(z,c):
    return c[:,0] + z@c[:,1:5].T + c[:,5]*z**3 + c[:,6]*z*z[...,nxt]

def coefficients(p):
    c = np.zeros((4,7)); g=p[4:8]
    c[:,0]=p[:4]; c[j,1+j]=g; c[j,1+nxt]=g*p[8]; c[j,1+prev]=g*p[9]; c[:,5]=g*p[10]; c[:,6]=g*p[11]
    return c

def rational(z,p):
    c=coefficients(p); q=z-p[13]*z[...,nxt]
    return forward(z,c) - c[:,5]*z**3*(p[12]*q**2)/(1+p[12]*q**2)

clean=np.max(abs(y),axis=1)<8  # Score only: visually separated EMI rows.
r=np.max(abs(x),axis=1); axis=np.count_nonzero(x,axis=1)==1
folds=[('axis',axis),('outer',r>=np.quantile(r,.8)),('central',r<=np.quantile(r,.2))]+[(f'rows{i}',np.arange(len(x))%5==i) for i in range(5)]
for name,test in folds:
    xx, yy = x[~test], y[~test]
    c=_fit_forward(xx,yy)
    residual=np.linalg.norm(forward(xx,c)-yy,axis=1); keep=residual<=6*np.median(residual)
    xx,yy=xx[keep],yy[keep]
    g=c[j,1+j]; p0=np.r_[c[:,0], g, np.median(c[j,1+nxt]/g), np.median(c[j,1+prev]/g), np.median(c[:,5]/g), np.median(c[:,6]/g)]
    p=least_squares(lambda p:(forward(xx,coefficients(p))-yy).ravel(),p0,gtol=1e-13).x
    # Profile signed pair combinations as possible cubic saturation coordinates.
    qs=[]
    for sign in (-1.,0.,1.):
        q=least_squares(lambda q:(rational(xx,np.r_[q,sign])-yy).ravel(),np.r_[p,0.],bounds=(np.r_[np.full(12,-np.inf),0.],np.inf),gtol=1e-13).x
        qs.append((np.linalg.norm(rational(xx,np.r_[q,sign])-yy),np.r_[q,sign]))
    q=min(qs,key=lambda v:v[0])[1]
    a,b=x[test&clean],y[test&clean]; z=_inverse(b,coefficients(p))
    zr=np.array([least_squares(lambda t:rational(t,q)-target,seed,gtol=1e-13).x for target,seed in zip(b,z)])
    print(name, 'max inverse independent/shared/rational:', *(float(np.max(np.abs(v-a))) for v in (_inverse(b,c),z,zr)), 'rational strength:',q[12])
print('study keys:', sorted(study))
