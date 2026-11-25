import importlib.util, itertools, pickle, time
import numpy as np
from scipy.optimize import lsq_linear
spec=importlib.util.spec_from_file_location('solver','/holdout_task/certified_l0_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
results={}; failures=[]
def check(label, a, ref=None):
    X,y,r,b,l=(a[k] for k in ('X','y','ridge','box','l0_penalty'))
    try:
        ans=m.solve_certified_box_l0(a); w=np.array(ans['weights']); rr=X@w-y
        v=.5*(rr@rr+r*(w@w))+l*np.count_nonzero(abs(w)>1e-9)
        gap=v-ans['certified_lower_bound']
        assert w.shape==(X.shape[1],) and np.all(abs(w)<=b) and np.isfinite(v)
        assert abs(v-ans['objective'])<=1e-10 and 0<=gap<=1e-5 and 1<=ans['nodes_explored']<=600
        if ref is not None: assert ans['certified_lower_bound']<=ref+2e-9 and ref-2e-9<=v<=ref+1e-5, (ref,v,ans)
        g,n=results.get(label,(0,0));results[label]=(max(g,gap),max(n,ans['nodes_explored']))
    except (AssertionError, RuntimeError) as e:
        failures.append((label,str(e)))
        if len(failures)<=5: np.savez('/tmp/l0_failure_'+str(len(failures))+'.npz',**a)
def args(X,y,r,b,l): return dict(X=X,y=y,ridge=r,box=b,l0_penalty=l)
with open('/holdout_task/public_box_l0_instances.pkl','rb') as f:
    for a in pickle.load(f): check('public',a)
p=15; r=.2; l=.7; t=1e-9; a0=1+r; y0=a0*t+np.sqrt(2*a0*l)
for b in [.2,1.5,1e-9,5e-10]:
    y=np.full(p,y0); z=np.clip(y0/a0,0,b); z0=min(t,z)
    ref=p*min(.5*((z0-y0)**2+r*z0*z0),.5*((z-y0)**2+r*z*z)+l)
    check('orthogonal tie and threshold',args(np.eye(p),y,r,b,l),ref)
for seed in range(60):
    rng=np.random.default_rng(seed); p=15; n=[15,24,42][seed%3]
    X=rng.normal(size=(n,p))/np.sqrt(n); y=rng.normal(size=n)
    if seed%4==1: X[:,8:]=X[:,:7]+.02*X[:,8:]
    if seed%4==2: X=.2*X+rng.normal(size=(n,3))@rng.normal(size=(3,p))/np.sqrt(n)
    if seed%4==3: X=np.repeat(np.eye(8),2,axis=1)[:,:p]; y=rng.normal(size=8)
    a=args(X,y,[.0001,.04,.35][seed%3],[.25,1.5,6][seed%3],[.05,.25,.7,.4,1.2][seed%5]); check('max-size regime '+str(seed%4),a)
for seed in range(8):
    rng=np.random.default_rng(1000+seed); n,p=9,7; X=rng.normal(size=(n,p)); y=rng.normal(size=n)*2
    r=.012; l=.3; b=[.12,1.5,1.01e-9,9e-10][seed%4]; A=np.vstack((X,np.sqrt(r)*np.eye(p))); Y=np.r_[y,np.zeros(p)]
    vals=[]
    for bits in itertools.product([False,True],repeat=p):
        d=np.where(bits,b,min(b,t)); w=np.clip(lsq_linear(A,Y,bounds=(-d,d),method='bvls',tol=1e-14).x,-d,d)
        rr=X@w-y; vals.append(.5*(rr@rr+r*(w@w))+l*np.count_nonzero(abs(w)>t))
    check('small independent reference',args(X,y,r,b,l),min(vals))
print('worst gap and node count by slice:',results,flush=True)
print('failures:',len(failures),failures[:20],flush=True)
