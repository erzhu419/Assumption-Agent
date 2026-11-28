import itertools
import pickle
import numpy as np
from scipy.optimize import least_squares

batch = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))
print('public studies:', len(batch), 'study keys:', sorted(batch[0]['study']))

def terms(x):
    out = {}
    for i,j in itertools.combinations(range(x.shape[1]), 2):
        for scale in (-2,-1,-.5,.5,1,2):
            d = x[:,i] + scale*x[:,j]
            for kind, v in [('square', d**2), ('cube', d**3), ('sin1', np.sin(d)), ('sin2', np.sin(2*d)), ('cos1', np.cos(d)), ('cos2', np.cos(2*d)), ('rational', d**3/(1+.7*d*d))]:
                out[kind,i,j,scale] = v
        out['product',i,j] = x[:,i]*x[:,j]
    for ijk in itertools.combinations(range(x.shape[1]), 3):
        out['triple',*ijk] = np.prod(x[:,ijk], axis=1)
    return out

def design(x,k,mode):
    q=x*x
    if mode == 'exponential':
        z=x*(-np.expm1(-k*q))
    elif mode == 'flex_power':
        z=x*q/(1+k[:-1]*q)**k[-1]
    elif mode == 'rational':
        z=x*q/(1+k*q)
    else:
        z=x*q/(1+k*(q+np.roll(q,1,axis=1)))
    return np.column_stack((np.ones(len(x)),x,z))

def fit(x,y,mode,term=None):
    p=np.full(x.shape[1],.8)
    low,high=.005,5.
    if mode=='flex_power':
        p=np.r_[p,1.]; low=np.r_[np.full(len(p)-1,.005),.25]; high=np.r_[np.full(len(p)-1,5.),3.]
    def A(a,k):
        b=design(a,k,mode)
        return np.column_stack((b,terms(a)[term])) if term is not None else b
    def calc(k):
        b=A(x,k); c=np.linalg.lstsq(b,y,rcond=None)[0]; return b@c-y,c
    k=least_squares(lambda k:calc(k)[0].ravel(),p,bounds=(low,high),max_nfev=80).x
    e,c=calc(k)
    return k,e,lambda a:A(a,k)@c

for entry in batch:
    panel=entry['study']['panel_b']; x,y=panel['u'],panel['v']; n=len(x)
    folds=[(np.arange(n)==i) for i in range(n)]
    outer=np.argsort(np.max(abs(x),axis=1))[-max(n//4,1):]
    tail=np.isin(np.arange(n),outer)
    for mode in ('rational','exponential','flex_power','coupled_rational','interaction'):
        picked=[]
        def run(m):
            xx,yy=x[~m],y[~m]; form='rational' if mode=='interaction' else mode
            k,e,pred=fit(xx,yy,form)
            if mode=='interaction':
                b=design(xx,k,form); q=np.linalg.qr(b,mode='reduced')[0]; h=(q*q).sum(axis=1)
                names, cols=zip(*terms(xx).items()); a=np.column_stack(cols); a-=q@(q.T@a)
                valid=(a*a).sum(axis=0)>1e-14
                a=a[:,valid]; a/=np.linalg.norm(a,axis=0)
                err=e[:,None,:]-a[:,:,None]*(a.T@e)[None,:,:]
                press=np.mean((err/np.maximum(1-h[:,None,None]-a[:,:,None]**2,1e-12))**2,axis=(0,2))
                base=np.mean((e/(1-h[:,None]))**2)
                if press.min()<.85*base:
                    term=[name for name,v in zip(names,valid) if v][int(press.argmin())]
                    picked.append(term); k,e,pred=fit(xx,yy,form,term)
            return pred(x[m])-y[m]
        _,e,_=fit(x,y,'rational' if mode=='interaction' else mode)
        loo=np.sqrt(np.mean(np.vstack([run(m) for m in folds])**2,axis=0))
        extra=np.sqrt(np.mean(run(tail)**2,axis=0))
        print(mode, 'fit RMSE',round(np.sqrt(np.mean(e**2)),6),'LOO RMSE',np.round(loo,6),'outer-quartile block RMSE',np.round(extra,6),'selected', {str(t):picked.count(t) for t in set(picked)})
