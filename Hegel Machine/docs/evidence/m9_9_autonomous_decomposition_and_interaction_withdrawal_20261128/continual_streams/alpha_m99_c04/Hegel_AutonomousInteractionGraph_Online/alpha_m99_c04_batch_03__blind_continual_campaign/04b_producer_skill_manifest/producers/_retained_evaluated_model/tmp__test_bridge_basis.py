import pickle
import numpy as np
from scipy.optimize import root

D = pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']
C = D['calibration_load_cells']; X,Y = C['true_channel_loads'],C['bridge_raw']
rng = np.random.default_rng(20)
folds = [np.r_[a,b+16] for a,b in zip(np.array_split(rng.permutation(16),8),np.array_split(rng.permutation(64),8))]
folds += [np.arange(i,i+4) for i in range(0,16,4)]
folds += [np.argsort(np.linalg.norm(X,axis=1))[-12:]]

def design(x, mode):
    p,n=np.roll(x,1,axis=1),np.roll(x,-1,axis=1)
    if mode=='ratio':
        return np.stack([np.ones_like(x), x+.15*p-.3*n-.22*x*n+.18*x**3],axis=-1)
    return np.stack([np.ones_like(x),x,p,n,x*n,x**3],axis=-1)

def fit(x,y,mode):
    a=design(x,mode)
    def solve(w):
        return np.stack([np.linalg.lstsq(a[:,j]*w[:,None],y[:,j]*w,rcond=None)[0] for j in range(4)])
    w=np.ones(len(x)); c=solve(w)
    for _ in range(30):
        e=np.linalg.norm(np.einsum('nif,if->ni',a,c)-y,axis=1)
        w=1/np.hypot(1, e/max(3e-4,1.5*np.median(e)))
        c=solve(w)
    good=e<max(.002,8*np.median(e))
    return solve(good.astype(float)),good

for mode in ['sparse','ratio']:
    c,good=fit(X,Y,mode); print(mode,'rejected',np.flatnonzero(~good).tolist(),'coefficients',np.round(c,7))
    worst=np.zeros((3,3))
    for i,te in enumerate(folds):
        tr=np.setdiff1d(np.arange(len(X)),te); c,_=fit(X[tr],Y[tr],mode)
        clean=te[good[te]]
        x,y=X[clean],Y[clean]
        yh=np.einsum('nif,if->ni',design(x,mode),c)
        z=np.array([root(lambda z: np.einsum('nif,if->ni',design(z[None,:],mode),c)[0]-r, np.clip(r,-2,2)).x for r in y])
        err=z-x
        scores=[np.max(abs(yh-y)),np.max(abs(err)),np.max(abs(err.sum(axis=1)))]
        s=0 if i<8 else 1 if i<12 else 2
        worst[s]=np.maximum(worst[s],scores)
    print('worst random, axis, radial: forward, inverse, total load',worst)
