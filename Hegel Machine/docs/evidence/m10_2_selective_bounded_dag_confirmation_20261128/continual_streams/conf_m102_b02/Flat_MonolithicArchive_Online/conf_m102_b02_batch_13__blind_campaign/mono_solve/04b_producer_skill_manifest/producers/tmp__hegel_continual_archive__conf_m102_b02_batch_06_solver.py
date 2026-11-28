"""Calibrate the observation coordinates and infer a shared trajectory/readout model."""
from itertools import combinations
import numpy as np
from scipy.optimize import least_squares


def _ls(a,b):
    return np.linalg.lstsq(a,b,rcond=1e-9)[0]


def _i(a):
    return np.column_stack((np.ones(len(a)),a))


def _pair(a):
    return a*np.roll(a,-1,axis=1)


def _cal(a):
    p=np.roll(a,-1,axis=1)
    return np.stack((np.ones_like(a),a,p,a*p,a**3),axis=-1)


def _calibrate(panel,x):
    d=_cal(panel['u']); v=panel['v']; keep=np.ones(len(v),bool)
    for _ in range(8):
        Co=np.array([_ls(d[keep,j],v[keep,j]) for j in range(4)])
        err=np.max(np.abs(np.einsum('nik,ik->ni',d,Co)-v),axis=1)
        take=err<.75
        if take.sum()<8 or np.array_equal(take,keep):
            break
        keep=take
    def fun(a):
        return np.einsum('ik,ik->i',_cal(a[None,:])[0],Co)
    return np.array([least_squares(lambda a:fun(a)-row,row).x for row in x]).reshape(-1,4)


def _operator(u,v,x,fn,init):
    def residual(p):
        d=fn(u,p)
        return (d@_ls(d,v)-v).ravel()
    p=least_squares(residual,init,max_nfev=140).x
    return fn(x,p)@_ls(fn(u,p),v)


def _cross(a,b):
    return np.column_stack([np.sum(a*np.roll(b,k,axis=1),axis=1) for k in range(4)])


def _ridge(a,b,x,alpha):
    sd=np.maximum(a.std(0),1e-6); sd[0]=1
    a=a/sd; pen=alpha*np.eye(a.shape[1]);pen[0,0]=0
    return (x/sd)@np.linalg.solve(a.T@a+pen,a.T@b)


def _response(d,y,x,fn=np.sin):
    beta=_ls(d,y); k=len(beta)
    def evaluate(p,a):
        return a@p[:k]+p[-1]*fn(a@p[k:-1])
    fits=[least_squares(lambda p:evaluate(p,d)-y,np.r_[beta,beta*scale,amp],max_nfev=125)
          for scale,amp in ((.65,.5),(1.,-.5),(2.,.5),(.65,-.5))]
    return evaluate(min(fits,key=lambda f:f.cost).x,x)


def _coords(study,x,bmode='sat',cmode='c'):
    z=_calibrate(study['panel_a'],x)
    b=study['panel_b']; c=study['panel_c']; u=c['u']
    if bmode=='sat':
        bf=lambda x,p:_i(np.c_[x,x/(1+np.exp(p[0])*x*x)])
        bp=[-.2]
    elif bmode=='sin':
        bf=lambda x,p:_i(np.c_[x,np.sin(np.exp(p[0])*x)])
        bp=[1.05]
    else:
        bf=lambda x,p:_i(x-p[0]*np.sin(p[1]*x))
        bp=[.4,2.85]
    w=_operator(b['u'],b['v'],z,bf,bp)
    q=_i(_pair(w))@_ls(_i(_pair(u)),c['v1'])
    if cmode=='cos':
        cf=lambda x,p:_i(x+p[0]*(1-np.cos(p[1]*x)))
        cp=[.72,1.2]
    elif cmode=='c4':
        cf=lambda x,p:_i(x+p[0]*x*x+p[1]*x**4)
        cp=[.53,-.064]
    else:
        cf=lambda x,p:_i(x+p[0]*x*x+p[1]*x**3)
        cp=[.45,0]
    r=_operator(u,c['v2'],w,cf,cp)
    return z,w,q,r


def _estimate(z,w,q,r,y,tr,ids,test,alpha=.03,fn=np.tanh,basis='broad',rank=3):
    base=np.c_[z,q,r]
    if basis=='broad':
        blocks=[_cross(a,b) for a,b in combinations((z,w,q,r),2)]
    elif basis=='sparse':
        blocks=[_cross(z,q),_cross(q,r)]
    else:
        blocks=[(z[:,:,None]*q[:,None,:]).reshape(len(z),16)]
    d=_i(np.column_stack([base,*blocks]))
    t=tr[ids]; mean=t.mean(0); modes=np.linalg.svd(t-mean,full_matrices=False)[2][:rank]
    scores=(t-mean)@modes.T
    predicted=_ridge(d[ids],scores,d[test],alpha)
    yh=_response(_i(np.c_[z[ids],q[ids],scores]),y[ids],_i(np.c_[z[test],q[test],predicted]),fn)
    return yh,predicted@modes+mean


def _predict(coords, alternate, y, tr, ids, test):
    # Average across interaction bases to reduce sensitivity to training coverage.
    y0,t0=_estimate(*coords,y,tr,ids,test)
    y1,_=_estimate(*coords,y,tr,ids,test,basis='sparse')
    y2,t2=_estimate(*alternate,y,tr,ids,test,alpha=.005,basis='outer',rank=1)
    return .25*y0+.50*y1+.25*y2, .60*t0+.40*t2


def solve_batch(study: dict) -> dict:
    train=study['train']; test=study['test']['x']; n=len(train['x'])
    x=np.vstack((train['x'],test))
    y,tr=_predict(_coords(study,x),_coords(study,x,'sine','cos'),
                  train['y'],train['traj'],np.arange(n),np.arange(n,n+len(test)))
    diag=np.array([y.mean(),y.std(),tr[:,-1].mean(),tr[:,-1].std()])
    return dict(y_hat=y,state_pred=tr,diagnostics_vec=diag)
