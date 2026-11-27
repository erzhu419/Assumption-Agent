import pickle
from itertools import combinations_with_replacement
import numpy as np
from scipy.optimize import least_squares

TERMS = [(j,) for j in range(4)] + [q for k in (2,3) for q in combinations_with_replacement(range(4),k)]

def basis(x):
    x = np.asarray(x)
    return np.column_stack([np.ones(len(x))] + [np.prod(x[:, q], axis=1) for q in TERMS])

def fit(x,y):
    a = basis(x); p = np.linalg.lstsq(a,y,rcond=None)[0]
    p = np.column_stack([least_squares(lambda z: (a@z-yi)/.003, pi, loss='cauchy', x_scale='jac').x for yi,pi in zip(y.T,p.T)])
    s = np.linalg.lstsq(a,y.sum(1),rcond=None)[0]
    return p,s

def invert(y,p,s,w=3.):
    y=np.asarray(y); shape=y.shape
    flat=y.reshape(-1,4)
    ans=[]
    for yi in flat:
        start=np.linalg.lstsq(p[1:5].T,yi-p[0],rcond=None)[0].clip(-1.9,1.9)
        def fun(z):
            a=basis(z[None,:])[0]; res=a@p-yi
            return np.r_[res-res.mean(), w*(a@s-yi.sum())]
        sol=least_squares(fun,start,bounds=(-2.5,2.5),max_nfev=70)
        ans.append(sol.x)
    return np.array(ans).reshape(shape)

if __name__=='__main__':
    i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0];d=i['dataset'];c=d['calibration_load_cells'];x=c['true_channel_loads'];y=c['bridge_raw']; p,s=fit(x,y)
    print('constant',p[0],s[0]);
    for term,b,si in zip(TERMS,p[1:],s[1:]):
        if max(abs(b))>.01 or abs(si)>.002: print(term,np.round(b,4),round(si,4))
    r=basis(x)@p-y
    print('cal forward residual by channel',np.quantile(abs(r),[.5,.9,.95,.99],axis=0)); print('largest raw residual rows',np.round(np.c_[x,y,r][np.argsort(np.linalg.norm(r,axis=1))[-8:]],3))
    folds=np.array_split(np.random.default_rng(98).permutation(len(x)),5)
    for w in (1.,5.,20.):
        out=np.empty_like(x)
        for ix in folds:
            keep=~np.isin(np.arange(len(x)),ix); out[ix]=invert(y[ix],*fit(x[keep],y[keep]),w=w)
        err=out.sum(1)-x.sum(1); print('inverse validation',w,np.quantile(abs(err),[.5,.9,.95,.99,1.]))
    for name,g in [('train',d['train']),('test',d['test']),('diagnostic',i['campaign_diagnostics'])]:
        raw=np.concatenate([g['past_channel_raw'],g['hold_channel_raw'][:,None]],axis=1); z=invert(raw,p,s,5.); e=(basis(z.reshape(-1,4))@p-raw.reshape(-1,4)); print(name,'load range',np.quantile(z,[0,.5,1]),'raw inversion residual',np.quantile(np.linalg.norm(e,axis=1),[.5,.9,.99,1.]))
