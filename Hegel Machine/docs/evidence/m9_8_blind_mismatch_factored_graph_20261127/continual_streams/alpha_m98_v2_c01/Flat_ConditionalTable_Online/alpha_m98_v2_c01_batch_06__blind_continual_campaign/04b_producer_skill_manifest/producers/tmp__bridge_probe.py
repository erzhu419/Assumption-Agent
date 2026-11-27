import itertools,pickle,numpy as np
from scipy.optimize import least_squares
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=item['dataset']; c=d['calibration_load_cells']; X=c['true_channel_loads']; Y=c['bridge_raw'];
# Three single-axis rows carry a large common offset, unlike any ordinary calibration trace.
clean=np.max(np.abs(Y),axis=1)<8

def powers(kind):
    terms=[()]+[(k,) for k in range(4)]
    if kind=='affine':return terms
    terms += [(k,k) for k in range(4)]
    if kind=='diag2':return terms
    if kind!='diag3':terms += list(itertools.combinations(range(4),2))
    if kind!='full2':terms += [(k,k,k) for k in range(4)]
    if kind=='full3':terms += [s for s in itertools.combinations_with_replacement(range(4),3) if len(set(s))>1]
    return terms

def design(x,terms):return np.stack([np.prod(x[...,s],axis=-1) if s else np.ones(x.shape[:-1]) for s in terms],axis=-1)

def forward(x,beta,terms):return design(x,terms)@beta

def inverse(y,beta,terms):
    y=np.asarray(y); x=(y-beta[0])@np.linalg.inv(beta[1:5]);
    for _ in range(18):
        jac=np.empty(x.shape[:-1]+(4,4))
        for k in range(4):
            inds=[i for i,s in enumerate(terms) if k in s]; reduced=[]
            for i in inds:
                s=list(terms[i]);s.remove(k);reduced.append(tuple(s))
            jac[..., :,k]=design(x,reduced)@(np.array([terms[i].count(k) for i in inds])[:,None]*beta[inds]);
        delta=np.linalg.solve(jac,(forward(x,beta,terms)-y)[...,None])[...,0];x-=np.clip(delta,-.7,.7)
        if np.max(abs(delta))<1e-10:break
    return x

def fit(x,y,terms):
    a=design(x,terms);return np.linalg.lstsq(a,y,rcond=None)[0]

if __name__=='__main__':
 print('excluded',np.flatnonzero(~clean));print('sweep-only residual diagnostic:')
 for idx in range(4):
    j=(np.count_nonzero(X,axis=1)==1)&(X[:,idx]!=0)&clean; print(idx,np.round(X[j,idx],3),np.round(Y[j],4))
 for kind in ('affine','diag2','full2','diag3','full3'):
    terms=powers(kind); pred=np.empty_like(X[clean]); raw=np.empty_like(pred);xx,yy=X[clean],Y[clean]
    for fold in range(7):
        j=np.arange(len(xx))%7==fold; b=fit(xx[~j],yy[~j],terms);raw[j]=forward(xx[j],b,terms)-yy[j];pred[j]=inverse(yy[j],b,terms)-xx[j]
    print(kind,'forward RMSE',np.sqrt(np.mean(raw**2)),'axial RMSE/max',np.sqrt(np.mean(pred.sum(1)**2)),np.max(abs(pred.sum(1))),'channel RMSE',np.sqrt(np.mean(pred**2,axis=0)))
    b=fit(xx,yy,terms)
    if kind in ('diag3','full3'):
        print('coefficients above 0.002',[(s,np.round(v,4).tolist()) for s,v in zip(terms,b) if np.max(abs(v))>.002]); print('specimen inverse residual',[np.max(abs(forward(inverse(d[g]['past_channel_raw'],b,terms),b,terms)-d[g]['past_channel_raw'])) for g in ('train','test')])
