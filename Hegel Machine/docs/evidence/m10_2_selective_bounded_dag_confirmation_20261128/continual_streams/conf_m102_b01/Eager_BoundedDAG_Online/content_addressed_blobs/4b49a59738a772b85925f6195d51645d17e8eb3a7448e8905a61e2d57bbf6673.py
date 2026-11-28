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
