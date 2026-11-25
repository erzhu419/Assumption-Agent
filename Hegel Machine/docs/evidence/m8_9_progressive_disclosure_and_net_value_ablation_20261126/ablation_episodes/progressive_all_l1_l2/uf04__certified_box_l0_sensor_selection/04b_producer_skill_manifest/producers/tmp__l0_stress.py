import sys, time, pickle
import numpy as np
from scipy.optimize import lsq_linear
sys.path.insert(0,'/holdout_task')
from certified_l0_solver import solve_certified_box_l0
rng=np.random.default_rng(7128)
def objective(X,y,r,M,f,w):
    return 0.5*np.sum((X@w-y)**2) + 0.5*r*np.sum(w*w) + f*np.count_nonzero(np.abs(w)>1e-9)
def analyze(name,X,y,r,M,f,ref=None):
    try:
        ans=solve_certified_box_l0(dict(X=X,y=y,ridge=r,box=M,l0_penalty=f)); w=np.array(ans['weights']); U=ans['objective']; L=ans['certified_lower_bound']; gap=U-L
        assert np.max(np.abs(w),initial=0)<=M and abs(U-objective(X,y,r,M,f,w))<1e-10 and 0<=gap<=1e-5 and 1<=ans['nodes_explored']<=600
        if ref is not None: assert L<=ref+1e-12 and U>=ref-1e-9 and U<=ref+1e-5, (L,ref,U)
        return ans['nodes_explored'],gap,name
    except Exception as e:
        print('FAIL',name,str(e),flush=True); return 0,0,name
with open('/holdout_task/public_box_l0_instances.pkl','rb') as f:
    public=pickle.load(f)
res=[analyze('public-'+str(i),*(c[k] for k in ('X','y','ridge','box','l0_penalty'))) for i,c in enumerate(public)]
for i in range(12):
    n,p=12,7; X=rng.normal(size=(n,p)); y=rng.normal(size=n)*2; r=.02 if i%2 else .5; M=.25 if i%3 else 1.5; f=.3
    A=np.vstack((X,np.sqrt(r)*np.eye(p))); Y=np.r_[y,np.zeros(p)]; ref=np.inf
    for mask in range(1<<p):
        limits=np.where([(mask>>j)&1 for j in range(p)],M,1e-9)
        w=lsq_linear(A,Y,bounds=(-limits,limits),tol=1e-12).x
        ref=min(ref,objective(X,y,r,M,f,w))
    res.append(analyze('oracle-'+str(i),X,y,r,M,f,ref))
for kind in ('dense','factor','pairs','near_pairs','undersampled','orthogonal_tie'):
    out=[]; start=time.monotonic()
    for s in range(12 if kind in ('dense','factor','near_pairs','undersampled') else 3):
        p=15; n=32; r=(.01,.12,.35)[s%3]; M=(.4,1.5,3.)[s%3]; f=(.18,.55,.85)[s%3]; ref=None
        if kind=='dense': X=rng.normal(size=(n,p))
        elif kind=='factor': X=rng.normal(size=(n,4))@rng.normal(size=(4,p))+.05*rng.normal(size=(n,p))
        elif kind=='pairs': X=np.eye(8)[:,np.arange(p)//2]
        elif kind=='near_pairs': X=np.linalg.qr(rng.normal(size=(n,8)))[0][:,np.arange(p)//2]+(.003,.03,.1)[s%3]*rng.normal(size=(n,p))
        elif kind=='undersampled': X=rng.normal(size=(7,p))
        else:
            X=np.eye(p); M=2.; f=.25; y=np.full(p,np.sqrt(2*f*(1+r))); ref=p*(.5*(y[0]-1e-9)**2+.5*r*1e-18)
        if kind!='orthogonal_tie':
            if kind=='pairs': y=np.full(8,max(.9, np.sqrt(2*f*(1+r))+.12))
            else: y=X@np.where(rng.uniform(size=p)<.45,rng.normal(size=p),0.)+rng.normal(size=X.shape[0])*1.2
        out.append(analyze(kind+'-'+str(s),X,y,r,M,f,ref))
    print(kind,'worst nodes',max(out),'worst gap',max(x[1] for x in out),'seconds',round(time.monotonic()-start,2),flush=True); res+=out
res.append(analyze('threshold-only',np.eye(3)*2e6,np.array([.001,-.01,.002]),.1,5e-10,2.,1/2*(.009**2+.001**2)+.1*5e-19))
print('worst passed nodes',max(res),'worst passed gap',max(x[1] for x in res))

assert all(n > 0 for n, _, _ in res), "at least one certification check failed"
