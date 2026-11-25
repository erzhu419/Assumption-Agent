import importlib.util,itertools,time
import numpy as np
from scipy.optimize import lsq_linear
p='/holdout_task/certified_l0_solver.py'
s=importlib.util.spec_from_file_location('solver',p); m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def check(label,X,y,ridge,box,l0,ref=None):
    t=time.monotonic(); r=m.solve_certified_box_l0(dict(X=X,y=y,ridge=ridge,box=box,l0_penalty=l0)); w=np.array(r['weights']); v=X@w-y; obj=.5*(v@v+ridge*(w@w))+l0*sum(abs(w)>1e-9); lb=r['certified_lower_bound'];gap=obj-lb
    assert np.max(abs(w),initial=0)<=box and r['objective']==obj and 0<=r['nodes_explored']<=600
    if ref is not None: assert lb <= ref+2e-12 and obj >= ref-2e-12,(label,lb,obj,ref)
    print(f'{label}: nodes={r["nodes_explored"]}, gap={gap:.5g}, time={time.monotonic()-t:.2f}s', 'FAIL' if gap>1e-5 else '',flush=True)
def reference(X,y,ridge,box,l0):
    p=X.shape[1]; A=np.vstack([X,np.sqrt(ridge)*np.eye(p)]); b=np.r_[y,np.zeros(p)]; best=np.inf
    for mask in itertools.product([False,True],repeat=p):
        ub=np.where(mask,box,1e-9); res=lsq_linear(A*ub,b,bounds=(-1,1),method='bvls',tol=1e-13); e=A@(res.x*ub)-b
        best=min(best,.5*(e@e)+l0*sum(mask))
    return best
for k in range(8):
    rng=np.random.default_rng(k); z=rng.normal(size=(9,3)); X=np.repeat(z,2,axis=1)+.025*rng.normal(size=(9,6)); y=rng.normal(size=9); box=.12+.07*k; ridge=.01+.05*k; l0=.025+.03*k
    check(f'oracle-{k}',X,y,ridge,box,l0,reference(X,y,ridge,box,l0))
for box in [.22,.6,1.5]:
    for target in [1.3,3.1,5.2]:
        X=np.ones((1,15)); y=np.array([target]); ridge=.015; l0=.15
        best=.5*max(0,target-15e-9)**2+.5*ridge*15e-18
        for k in range(1,16):
            t=1e-9; lo=0 if k==15 else t*(k+ridge)/(k+ridge); a=min(box,(target-(15-k)*lo)/(k+ridge)); b=min(t,(target-k*a)/(15-k+ridge)) if k<15 else 0
            best=min(best,.5*((k*a+(15-k)*b-target)**2+ridge*(k*a*a+(15-k)*b*b))+k*l0)
        check(f'identical-{box}-{target}',X,y,ridge,box,l0,best)
for k in range(8):
    rng=np.random.default_rng(k+20); z=rng.normal(size=(20,3)); X=np.repeat(z,5,axis=1)+[0,.001,.01,.04][k%4]*rng.normal(size=(20,15)); X/=np.linalg.norm(X,axis=0); y=z@rng.normal(size=3)*.45+.08*rng.normal(size=20)
    check(f'five-way-{k}',X,y,.005,.55,.12)
