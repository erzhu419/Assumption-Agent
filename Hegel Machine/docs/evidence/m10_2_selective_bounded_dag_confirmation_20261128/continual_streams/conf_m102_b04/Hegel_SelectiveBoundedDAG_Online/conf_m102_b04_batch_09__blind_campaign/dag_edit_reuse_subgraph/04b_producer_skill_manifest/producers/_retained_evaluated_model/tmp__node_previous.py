import numpy as np
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import minimize, minimize_scalar

def aff(q): return np.c_[np.ones(len(q)),q]

def representation(st,x):
    c=st['panel_c']; b=st['panel_b']
    def fit(key, fn, bounds):
        u,v=c['u'],c[key]
        def solve(p):
            d=fn(u,p); w=np.linalg.lstsq(d,v,rcond=None)[0]; return np.mean((d@w-v)**2),w
        p=minimize_scalar(lambda p:solve(p)[0],bounds=bounds,method='bounded').x
        return fn(x,p)@solve(p)[1]
    v=fit('v1',lambda q,p:np.c_[aff(np.arcsinh(p*q)),q**3],(.1,8))
    w=fit('v2',lambda q,p:np.c_[aff(q),np.sqrt(1+p*q*q)-1],(.1,3))
    z=aff(x)@np.linalg.lstsq(aff(b['u']),b['v'],rcond=None)[0]
    return [x,v,w,z]

def predict(q,Y):
    n=len(Y); m=len(q[0]); y0=Y.mean(0); sc=np.maximum(Y.std(0),1e-7); yy=(Y-y0)/sc
    # Estimate separate variance components for interpretable coordinate groups.
    Q=[z/np.maximum(z[:n].std(0),1e-7) for z in q]
    d=Q[0].shape[1]; mats=[np.ones((m,n))]
    for z in Q:
        dif=(z[:,None,:]-z[None,:n,:])**2
        lin=z@z[:n].T/d
        mats.extend([lin,lin**2])
        for a in [.08,.35,1.4]:
            per=np.exp(-a*dif); p=per.sum(2)
            mats.extend([p/d,(p*p-(per**2).sum(2))/(d*(d-1)),np.exp(-a*dif.mean(2))])
    M=np.ascontiguousarray(mats); A=np.ascontiguousarray(np.concatenate([M[:,:n,:],np.eye(n)[None]],axis=0)); B=A.reshape(len(A),-1)
    def loss(p):
        a=np.exp(p); k=(a@B).reshape(n,n); cf=cho_factor(k); h=cho_solve(cf,yy); inv=cho_solve(cf,np.eye(n)); g=(inv-h@h.T/yy.shape[1])*.5
        return .5*np.sum(yy*h)/yy.shape[1]+np.log(cf[0].diagonal()).sum()+.012*np.sum((p+3)**2), a*(B@g.ravel())+.024*(p+3)
    par=minimize(loss,np.r_[0,np.full(len(A)-2,-4.),-7.],jac=True,method='L-BFGS-B',bounds=[(-14,4)]*(len(A)-1)+[(-16,0)],options={'maxiter':110}).x
    a=np.exp(par); cross=np.tensordot(a[:-1],M,axes=1); return y0+sc*(cross[n:]@cho_solve(cho_factor((a@B).reshape(n,n)),yy))

def run_node(inputs):
    st=inputs['__input__']; latent=inputs['node_00']; q=representation(st,np.cbrt(np.r_[latent['cubic_train'],latent['cubic_test']]))
    y=predict(q,np.asarray(st['train']['y'])[:,None])[:,0]; t=np.asarray(st['train']['traj']); u,_,v=np.linalg.svd(t,full_matrices=False)
    p=predict(q,t@v[:1].T)@v[:1]+predict(q,t@v[1:2].T)@v[1:2]
    return dict(y_hat=y,state_pred=p,diagnostics_vec=np.array([y.mean(),y.std(),p[:,-1].mean(),p[:,-1].std()]))
