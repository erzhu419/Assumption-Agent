exec(open('/tmp/rheo_probe.py').read().split('for kind in')[0])
from scipy.optimize import minimize


def source_predict(g,z,tau,w):
    return s._design(g,z,b,dt,t,tau)@w


def basis(g,z,mode):
    l,u=g['lab_id'],g['stratum']
    a=(l[:,None]==np.unique(tr['lab_id'])).astype(float)
    c=(u[:,None]==np.unique(tr['stratum'])).astype(float)
    cell=(a[:,:,None]*c[:,None,:]).reshape(len(l),-1)
    main=np.column_stack((np.ones(len(l)),z))
    if mode=='global': return main,np.array([.10]+[.08]*4)
    if mode=='lab': return np.column_stack((main,a)),np.array([.10]+[.08]*4+[.09]*a.shape[1])
    if mode=='cell': return np.column_stack((main,cell)),np.array([.10]+[.08]*4+[.12]*cell.shape[1])
    return np.column_stack((main,a,c,cell)),np.array([.10]+[.08]*4+[.065]*(a.shape[1]+c.shape[1])+[.10]*cell.shape[1])

bt,bw,_=s._fit(tr,x[0],b,dt,t)
sc=np.r_[np.full(len(t),.009),.014]; h=np.r_[t,1.]

def fit(ix,mode,nuisance=.025):
    z=x[2][ix]; g={k:v[ix] for k,v in pi.items()}
    D,prior=basis(g,z,mode); n=len(z)
    U=np.column_stack((D,np.eye(n)*nuisance))
    U[:,:len(prior)]*=prior
    Q=(U[:,None,:]*(h/sc)[None,:,None]).reshape(-1,U.shape[1]); y=s._obs(g)
    y=(y-(bw[2]+z@bw[3:])[:,None]*h)/sc
    reg=np.diag(np.r_[[1e-7,1e-7],np.ones(U.shape[1])])
    def obj(p,ret=False):
        tau=np.sort(np.exp(p)); A=s._design(g,z,b,dt,t,tau)[:,:,:2]/sc[None,:,None]
        A=np.column_stack((A.reshape(-1,2),Q)); yy=y.ravel()
        beta=np.linalg.lstsq(np.vstack((A,np.sqrt(reg))),np.r_[yy,np.zeros(len(reg))],rcond=None)[0]
        for j in range(11):
            r=A@beta-yy; w=1/(1+(r/2.5)**2)
            beta=np.linalg.solve(A.T@(w[:,None]*A)+reg, A.T@(w*yy))
        if ret: return tau,beta[:2],beta[2:2+len(prior)]*prior
        return np.sum(6.25*np.log1p(((A@beta-yy)/2.5)**2))+np.sum((np.sqrt(reg)@beta)**2)
    bounds=[(np.log(.05001),np.log(24.999))]*2
    p=min((minimize(obj,np.log(start),bounds=bounds,method='L-BFGS-B') for start in [bt,[.65,7.]]),key=lambda r:r.fun).x
    return obj(p,True)

for mode in ['baseline','global','lab','cell','hier']:
    for scheme in ['coupon','cell','lab']:
        tags=np.arange(len(pi['lab_id'])) if scheme=='coupon' else pi['lab_id']*(3 if scheme=='cell' else 1)+(pi['stratum'] if scheme=='cell' else 0)
        pred=np.zeros_like(s._obs(pi))
        for k in np.unique(tags):
            ix,js=tags!=k,tags==k
            g={key:v[js] for key,v in pi.items()}; z=x[2][js]
            if mode=='baseline':
                a,w,_=s._fit({key:v[ix] for key,v in pi.items()},x[2][ix],b,dt,t)
                pred[js]=source_predict(g,z,a,w)
            else:
                a,w,c=fit(ix,mode); D,_=basis(g,z,mode)
                pred[js]=s._design(g,z,b,dt,t,a)[:,:,:2]@w+(bw[2]+z@bw[3:]+D@c)[:,None]*h
        r=abs(pred-s._obs(pi)); c=r[:,:-1]; v=r[:,-1]
        print(mode,scheme,'mean',np.round([c.mean(),v.mean()],4),'p90',np.round([np.quantile(c,.9),np.quantile(v,.9)],4),'max cell mae',round(max(c[(pi['lab_id']*3+pi['stratum'])==i].mean() for i in np.unique(pi['lab_id']*3+pi['stratum'])),4),flush=True)
    if mode!='baseline': print('full fit',fit(np.ones(len(pi['lab_id']),bool),mode),flush=True)
