import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize, least_squares
from scipy.spatial.distance import cdist
from itertools import combinations_with_replacement

def affine(panel, key, robust=False):
    u, v = panel['u'], panel[key]
    f = np.column_stack((np.ones(len(u)),u)); ids=np.arange(len(u))
    for _ in range(7 if robust else 1):
        c=np.linalg.lstsq(f[ids],v[ids],rcond=None)[0]
        e=np.linalg.norm(f@c-v,axis=1)
        if robust: ids=np.flatnonzero(e<max(.001,4*np.median(e)))
    return c

def apply(c,q): return q@c[1:]+c[0]

def temporal(t,z):
    def basis(p):
        k,d,w=p
        return np.column_stack((-np.expm1(-k*t),np.exp(-d*t)*np.sin(w*t)))
    def res(p):
        f=basis(p);return (f@np.linalg.lstsq(f,z.T,rcond=None)[0]-z.T).ravel()
    return np.linalg.qr(basis(least_squares(res,[.5,.2,1.1],bounds=([.01,0,.4],[4,2,2])).x))[0]

def blocks(s,x,n,mode):
    a=affine(s['panel_a'],'v',True); b=affine(s['panel_b'],'v'); c=affine(s['panel_c'],'v1')
    q=apply(a,x); p=apply(b,q); v=apply(c,q)
    def scale(u): return (u-u[:n].mean(0))/np.maximum(u[:n].std(0),1e-6)
    def gram(u): return u@u.T
    ans=[np.ones((len(x),len(x)))]; names=['const']
    def add(k,name):
        ans.append(k);names.append(name)
    for key,u in [('A',q),('raw',x)]:
        g=gram(scale(u))/4
        for j in range(1,5 if mode!='smooth' else 3): add(g**j,key+str(j))
        d=cdist(scale(u),scale(u),'sqeuclidean')
        for r in (.8,1.6,3.2,6.4): add(np.exp(-d/(2*r*r)),key+'rbf'+str(r))
        d=np.sqrt(5*d)/2
        add((1+d+d*d/3)*np.exp(-d),key+'matern')
    if mode in ('panel','sparse'):
        for key,u in [('A',q),('BA',p),('C',v),('raw',x)]:
            g=gram(scale(u))/4
            for k in (.6,1.3):
                for fn in (np.sin,np.tanh,np.cos):
                    f=fn(k*u); f-=f[:n].mean(0); h=gram(f)
                    add(h,key+fn.__name__+str(k)); add(h*g,key+fn.__name__+'cross'+str(k));add(h*h,key+fn.__name__+'sq'+str(k))
    if mode=='sparse':
        for key,u in [('A',q),('BA',p),('C',v)]:
            u=scale(u)
            for j,f in enumerate(u.T):
                for k in (3,4): add(np.outer(f**k,f**k),key+str(j)+'p'+str(k))
    return np.array(ans),names

def predict(ks,y,prior=.02):
    n=len(y); k=ks[:,:n,:n]; l=ks[:,n:,:n]; h=np.maximum(np.diagonal(k,axis1=1,axis2=2).mean(1),1e-12)
    I=np.eye(n); k=np.concatenate((k/h[:,None,None],I[None])); l=np.concatenate((l/h[:,None,None],np.zeros_like(l[:1]))); mu=y.mean(0); sd=np.maximum(y.std(0),1e-10); z=(y-mu)/sd
    if z.ndim==1:z=z[:,None]
    m=z.shape[1]; kf=k.reshape(len(k),-1); lf=l.reshape(len(l),-1)
    p0=np.full(len(k),-5.);p0[-1]=-6
    def fun(p):
        w=np.exp(p); C=cho_factor((w@kf).reshape(n,n),check_finite=False)
        a=cho_solve(C,z,check_finite=False); inv=cho_solve(C,I,check_finite=False)
        return .5*np.sum(a*z)+m*np.log(np.diag(C[0])).sum()+.5*prior*np.sum((p-p0)**2), .5*w*(kf@(m*inv-a@a.T).ravel())+prior*(p-p0)
    pp=minimize(fun,p0,jac=True,method='L-BFGS-B',bounds=[(-16,4)]*len(k),options={'maxiter':180,'ftol':1e-9}).x
    w=np.exp(pp); C=cho_factor((w@kf).reshape(n,n),check_finite=False); a=cho_solve(C,z,check_finite=False)
    pred=(w@lf).reshape(-1,n)@a*sd+mu
    return pred,w

if __name__=='__main__':
    import pickle
    s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; z=s['train']['traj']; t=s['eval_times']
    for key in ('panel_a','panel_b'):
        a=affine(s[key],'v',key=='panel_a');p=s[key];e=np.linalg.norm(apply(a,p['u'])-p['v'],axis=1);print(key,np.round(a,4),'residual quantiles',np.quantile(e,[.25,.5,.8,.95,1]),flush=True)
    folds=[np.arange(len(x))]+[np.random.default_rng(k+135).permutation(len(x)) for k in range(6)]
    for mode in ('smooth','poly','panel','sparse'):
        vals=[]
        for ids in folds:
            i,j=ids[:56],ids[56:]; Q=temporal(t,z[i]); K,names=blocks(s,x[ids],56,mode)
            yt,w=predict(K,y[i]); at,_=predict(K,z[i]@Q); zt=at@Q.T
            vals.append([abs(yt[:,0]-y[j]).mean(),abs(zt-z[j]).mean(),abs(zt[:,-1]-z[j,-1]).mean()]);
            if len(vals)==1: print(mode,'target weights',sorted(zip(np.round(w,5),names+['noise']),reverse=True)[:8],flush=True)
        print(mode,'first/mean/worst',np.round(vals[0],6),np.round(np.mean(vals,0),6),np.round(np.max(vals,0),6),flush=True)
