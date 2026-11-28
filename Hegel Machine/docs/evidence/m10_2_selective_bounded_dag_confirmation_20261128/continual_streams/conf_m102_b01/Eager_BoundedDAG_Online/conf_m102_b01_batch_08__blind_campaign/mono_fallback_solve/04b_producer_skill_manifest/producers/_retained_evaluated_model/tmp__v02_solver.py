import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
from scipy.linalg import cho_factor,cho_solve
from itertools import combinations_with_replacement

def _ls(a,b): return np.linalg.lstsq(a,b,rcond=None)[0]
def _aff(x): return np.c_[np.ones(len(x)),x]
def _roll(x): return np.roll(x,-1,axis=1)
def _cal(s,x):
    u,v=(s['panel_a'][k] for k in ('u','v')); qs=[]
    for j in range(4):
        a,b=u[:,j],u[:,(j+1)%4]; f=np.c_[np.ones(len(a)),a,b,a*b,a**3]
        t=v[:,j]; q=least_squares(lambda q:f@q-t,_ls(f,t),loss='cauchy',f_scale=.015).x
        for _ in range(3):
            e=abs(f@q-t); k=e<max(.001,7*np.median(e)); q=_ls(f[k],t[k])
        qs.append(q)
    q=np.array(qs).T
    def fun(p): return q[0]+q[1]*p+q[2]*np.roll(p,-1)+q[3]*p*np.roll(p,-1)+q[4]*p**3
    return np.array([least_squares(lambda p:fun(p)-r,r).x for r in x])

def _routes(s,x):
    z=_cal(s,x); u,v=(s['panel_b'][k] for k in ('u','v')); c=s['panel_c']; cu=c['u'];
    def f1(p): return _aff(p*_roll(p))
    def f2(p): return _aff(np.c_[p,p*p,np.exp(-.75*p*p)])
    def b1(p): return _aff(np.c_[p,p**3,p*_roll(p),np.tanh(1.25*p**3)])
    def b2(p): return _aff(np.c_[p,p**3,np.cbrt(p),p*_roll(p),np.sinh(.4*p**3)])
    return [x,z,f1(z)@_ls(f1(cu),c['v1']),f2(z)@_ls(f2(cu),c['v2']),b1(z)@_ls(b1(u),v),b2(z)@_ls(b2(u),v),z**3]

def _kernels(routes,n):
    def norm(w): return (w-w[:n].mean(0))/np.maximum(w[:n].std(0),1e-7)
    def dot(w): return w@w[:n].T/w.shape[1]
    R=[norm(w) for w in routes]; a,x,c,d,b,b2,p=R; sets=[a,x,b,b2,d,np.c_[x,b],np.c_[x,b,b2,c,d,p]]; K=[]
    for w in R:
        K.append(dot(w))
        for g in (.6,1.7):
            h=norm(np.arcsinh(g*w)); h=np.c_[w,h]; K.append(dot(h));
            f=norm(np.column_stack([h[:,i]*h[:,j] for i,j in combinations_with_replacement(range(h.shape[1]),2)])); K.append(dot(h)+dot(f))
    for w in sets:
        D=cdist(w,w[:n],'sqeuclidean')/w.shape[1]
        for l in (.55,1.1,2.2,4.4):
            G=np.sqrt(3*D)/l; K.extend([np.exp(-.5*D/l**2),(1+G)*np.exp(-G),dot(w)+np.exp(-.5*D/l**2)])
    return K

def _stack(ks,t):
    n=len(t); pred=[]; loo=[]; t=np.asarray(t); t=t[:,None] if t.ndim==1 else t
    for k in ks:
        for r in (1e-4,.002,.015,.08):
            a=k[:n]+5+np.eye(n)*r
            try:
                f=cho_factor(a); b=cho_solve(f,t); h=np.diag(cho_solve(f,np.eye(n))); e=b/h[:,None]
            except np.linalg.LinAlgError: continue
            loo.append(np.mean(abs(e))); pred.append((k[n:]+5)@b)
    losses=np.array(loo); ix=np.argsort(losses)[:16]; w=np.exp(-(losses[ix]-losses[ix[0]])/max(.05*losses[ix[0]],1e-9));return np.average(np.array(pred)[ix],axis=0,weights=w)

def solve_batch(study):
    tr=study['train']; n=len(tr['x']); x=np.r_[tr['x'],study['test']['x']]; ks=_kernels(_routes(study,x),n)
    y=_stack(ks,tr['y'])[:,0]; p=_stack(ks,tr['traj']); tail=p[:,-1]
    return {'y_hat':y,'state_pred':p,'diagnostics_vec':np.array([y.mean(),y.std(),tail.mean(),tail.std()])}

from scipy.optimize import minimize


def _panel_b_candidates(study, z):
    panel = study['panel_b']
    u, v = panel['u'], panel['v']
    def signed(w, g):
        return np.sign(w)*np.abs(w)**g
    funcs = [
        lambda w,g: np.arcsinh(g*w**3),
        lambda w,g: np.tanh(g*w)**3,
        lambda w,g: np.sin(g*w)**3,
        lambda w,g: w**3/(1+g*w*w),
        lambda w,g: w**3/(1+g*w*w)**1.5,
        lambda w,g: np.arctan(g*w**3),
        lambda w,g: np.tanh(g*w**3),
        lambda w,g: np.arctan(g*w)**3,
        lambda w,g: signed(w,g),
    ]
    fits = []
    for j, fn in enumerate(funcs):
        for mode in range(3):
            def design(w,g):
                return _aff(np.c_[fn(w,g), *[(),(w,),(w,w**3)][mode]])
            def residual(g):
                F=design(u,g[0]); q=_ls(F,v)
                h=np.sum(F*np.linalg.pinv(F).T,axis=1)
                return ((F@q-v)/np.maximum(1-h,1e-5)[:,None]).ravel()
            opt=least_squares(residual,[1.25 if j!=8 else 3.],bounds=(.06,6.),max_nfev=35)
            F=design(u,opt.x[0]); q=_ls(F,v)
            fits.append((np.mean(abs(opt.fun)),design(z,opt.x[0])@q,fn(z,opt.x[0])))
    fits.sort(key=lambda r:r[0]); return fits[:3]


def _adaptive(routes, targets, head):
    n=len(targets); target=np.asarray(targets); target=target[:,None] if target.ndim==1 else target
    center=target.mean(0); vals=target-center
    if vals.shape[1]>1:
        _,_,V=np.linalg.svd(vals,full_matrices=False); y=vals@V[0]
    else: y=vals[:,0]
    scale=max(np.std(y),1e-9); y=y/scale
    def norm(w):
        return np.clip((w-w[:n].mean(0))/np.maximum(w[:n].std(0),1e-7),-12,12)
    eye=np.eye(n); out=[]
    for idx,w in enumerate(routes):
        w=norm(w); dim=w.shape[1];
        tt=w[:n]; deltas=(w[:,None,:]-tt[None,:,:])**2
        def kernel(p,all_rows=False,deriv=False):
            ds=deltas if all_rows else deltas[:n]; weights=np.exp(p[:dim])/dim
            D=ds@weights; s=np.sqrt(5*np.maximum(D,0)); e=np.exp(-s)
            m=(1+s+s*s/3)*e
            r=np.exp(-D/2)
            K=.7*m+.3*r+.05*(w if all_rows else tt)@tt.T/dim+.2
            if not deriv: return K
            g=-.7*(5/6)*(1+s)*e-.15*r
            return K,g
        def stats(p,gradient=True):
            K,g=kernel(p,deriv=True); noise=np.exp(p[-1]); K=K+noise*eye
            f=cho_factor(K,check_finite=False); inv=cho_solve(f,eye,check_finite=False); a=inv@y
            if not gradient:return f, np.mean(abs(a/np.diag(inv)))
            t=max(y@a/n,1e-12)
            diff=.5*(inv-np.outer(a,a)/t)
            obj=.5*n*np.log(t)+np.log(np.diag(f[0])).sum()+.24*np.sum(p[:dim]**2)
            grad=np.einsum('ij,ijk->k',diff*g,deltas[:n])*np.exp(p[:dim])/dim+.48*p[:dim]
            return obj,np.r_[grad,noise*np.trace(diff)]
        for start in (-1.,-3.):
            p=np.r_[np.full(dim,start),-8.]
            fit=minimize(stats,p,jac=True,method='L-BFGS-B',bounds=[(-8,4)]*dim+[(-13,-1.5)],options={'maxiter':90})
            for q in (fit.x,p):
                f,loo=stats(q,False)
                pred=kernel(q,True)[n:]@cho_solve(f,vals,check_finite=False)+center
                out.append((loo,pred))
    for w in routes:
        w=norm(w); D=cdist(w,w[:n],'sqeuclidean')/w.shape[1];
        # Explicit pair and three-coordinate interactions, with a shrinkage prior.
        base=w@w[:n].T/w.shape[1]
        for kind in range(2):
            for l in (.7,1.25,2.25,4.):
                if kind==0:
                    a=np.sqrt(5*D)/l; K=(1+a+a*a/3)*np.exp(-a)+.09*base
                else:
                    # Additive component permits unequal coordinate contributions.
                    h=np.exp(-.5*(w[:,None,:]-w[None,:n,:])**2/l**2).mean(2)
                    K=np.exp(-.5*D/l**2)+.3*h+.04*base+.04*base**2+.015*base**3
                for ridge in (3e-5,.0003,.002,.015):
                    F=cho_factor(K[:n]+.5+ridge*eye,check_finite=False); inv=cho_solve(F,eye,check_finite=False)
                    loo=np.mean(abs((inv@y)/np.diag(inv)))
                    pred=(K[n:]+.5)@(inv@vals)+center
                    out.append((loo,pred))
    out.sort(key=lambda a:a[0]); picks=out[:18]; best=picks[0][0]; weights=[np.exp(-(r-best)/max(.085*best,1e-9)) for r,_ in picks]
    return np.average(np.array([p for _,p in picks]),axis=0,weights=weights),best*scale


_ensemble_baseline=solve_batch

def _updated(study):
    tr=study['train']; n=len(tr['x']); x=np.r_[tr['x'],study['test']['x']]; routes=_routes(study,x)
    a,z,c,d,b,f,p=routes
    fits=_panel_b_candidates(study,z)
    candidates=[b,f,*[r[1] for r in fits],*[r[2] for r in fits],np.c_[z,b],np.c_[z,fits[0][1]],np.c_[fits[0][1],c,d],z,a]
    K=_kernels(routes,n)
    y0=_stack(K,tr['y'])[:,0]; p0=_stack(K,tr['traj'])
    y,e1=_adaptive(candidates,tr['y'],0); t,e2=_adaptive(candidates,tr['traj'],1)
    y=.85*y[:,0]+.15*y0; t=.70*t+.30*p0; tail=t[:,-1]
    return {'y_hat':y,'state_pred':t,'diagnostics_vec':np.array([y.mean(),y.std(),tail.mean(),tail.std()])}

solve_batch=_updated


def solve_batch(study):
    old=_ensemble_baseline(study)
    new=_updated(study)
    y=new['y_hat']
    t=old['state_pred']
    return {'y_hat':y,'state_pred':t,'diagnostics_vec':np.array([y.mean(),y.std(),t[:,-1].mean(),t[:,-1].std()])}
