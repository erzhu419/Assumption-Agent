import sys, pickle, time
sys.path.insert(0, '/holdout_task')
import numpy as np
from itertools import combinations
from scipy.optimize import lsq_linear
from certified_box_l0 import solve_certified_box_l0


def verify(c, name):
    X,y = c['X'],c['y']; p=X.shape[1]; M,pen,reg=(c[k] for k in ('box','l0_penalty','ridge'))
    t=time.perf_counter(); out=solve_certified_box_l0(c); dt=time.perf_counter()-t
    w=np.array(out['weights']); obj=lambda w: .5*np.linalg.norm(X @ w-y)**2+.5*reg*(w@w)+pen*np.count_nonzero(abs(w)>1e-10)
    assert w.shape==(p,) and max(abs(w),default=0)<=M and abs(obj(w)-out['objective'])<1e-7
    assert 0<=out['objective']-out['certified_lower_bound']<=c['gap_tol']
    leaves=out['certificate_leaves']; assert 1<=len(leaves)<=c['max_leaves']
    masks=np.arange(1<<p); membership=np.zeros(len(masks),int); claims=np.full(len(masks),np.nan)
    for leaf in leaves:
        z,o=map(leaf.get,('fixed_zero','fixed_nonzero'))
        assert len(z)+len(o)==len(set(z+o)) and all(0<=j<p for j in z+o)
        ids=((masks&sum(1<<j for j in z))==0)&((masks&sum(1<<j for j in o))==sum(1<<j for j in o))
        membership[ids]+=1; claims[ids]=leaf['leaf_lower_bound']
    assert np.all(membership==1) and out['certified_lower_bound']<=min(claims)+1e-8
    A=np.vstack((X,np.sqrt(reg)*np.eye(p))); b=np.r_[y,np.zeros(p)]; exact=.5*(b@b); refinements=0
    assert claims[0]<=exact+1e-8
    for k in range(1,p+1):
        if k*pen>=max(exact,max(claims))+1e-8: break
        for j in combinations(range(p),k):
            mask=sum(1<<i for i in j); B=A[:,j]
            u=np.linalg.lstsq(B,b,rcond=None)[0]; v=.5*np.linalg.norm(B@u-b)**2+k*pen
            if max(abs(u))>M and v<max(claims[mask],exact):
                u=lsq_linear(B,b,bounds=(-M,M),method='bvls',tol=1e-14).x
                v=.5*np.linalg.norm(B@u-b)**2+k*pen
                refinements+=1
            assert claims[mask]<=v+1e-8,(name,mask,claims[mask],v)
            exact=min(exact,v)
    print(name,'time',round(dt,3),'gap',out['objective']-out['certified_lower_bound'],'nleaves',len(leaves),'box reference fits',refinements,flush=True)
    assert abs(exact-out['objective'])<=c['gap_tol']

for i,c in enumerate(pickle.load(open('/holdout_task/public_box_l0_cases.pkl','rb'))['cases']): verify(c,f'public {i}')
rng=np.random.default_rng(19)
for i,(p,noise,ridge,box,pen,cap) in enumerate([(15,0.,0.,.4,.16,512),(15,.002,0.,.12,.04,1),(16,.02,.02,.2,.04,2),(14,.001,0.,1.3,.001,512),(12,0.,0.,.25,.03,512)]):
    n=38
    X = rng.normal(size=(n,4)) @ rng.normal(size=(4,p)) + noise*rng.normal(size=(n,p))
    y = X@rng.uniform(-1,1,p)*.6 + rng.normal(size=n)*.05
    if i==4: X[:,0]=0; X[:,2]=X[:,1]
    verify(dict(X=X,y=y,ridge=ridge,box=box,l0_penalty=pen,gap_tol=1e-6,max_leaves=cap),f'stress {i}')
