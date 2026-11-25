import importlib.util,pickle,itertools
import numpy as np
from scipy.optimize import lsq_linear
root='/holdout_task/'
spec=importlib.util.spec_from_file_location('solver',root+'certified_l0_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def solve(label,a,ref=None):
    r=m.solve_certified_box_l0(a); X=np.asarray(a['X']); w=np.asarray(r['weights']); e=X@w-a['y']
    j=.5*(e@e+a['ridge']*(w@w))+a['l0_penalty']*np.count_nonzero(abs(w)>1e-9)
    lb=r['certified_lower_bound']; gap=j-lb
    assert set(r)=={'weights','objective','certified_lower_bound','nodes_explored'}
    assert w.shape==(X.shape[1],) and np.max(abs(w))<=a['box'] and abs(j-r['objective'])<1e-12 and 0<=r['nodes_explored']<=600
    if ref is not None: assert lb<=ref+2e-10 and j>=ref-2e-10,(label,j,lb,ref)
    print(f'{label}: gap={gap:.9g} nodes={r["nodes_explored"]} {"PASS" if 0<=gap<=1e-5 else "FAIL"}',flush=True)
    return lb,j

def reference(a):
    X=a['X']; p=X.shape[1]; A=np.vstack((X,np.sqrt(a['ridge'])*np.eye(p))); b=np.r_[a['y'],np.zeros(p)]
    best=np.inf
    for mask in itertools.product((0,1),repeat=p):
        radius=np.where(mask,a['box'],1e-9); D=A*radius
        fit=lsq_linear(D,b,bounds=(-1,1),method='bvls',tol=1e-14)
        e=D@fit.x-b; best=min(best,.5*(e@e)+a['l0_penalty']*sum(mask))
    return best

for i,a in enumerate(pickle.load(open(root+'public_box_l0_instances.pkl','rb'))):
    lo,hi=solve(f'public-{i}',a)
    ix=np.arange(a['X'].shape[1])[::-1]; signs=1-2*(ix%2)
    lb,j=solve(f'public-{i}-signed-reversal',dict(a,X=a['X'][::-1,ix]*signs,y=a['y'][::-1]))
    assert max(lo,lb)<=min(hi,j)+2e-10
    for box,penalty in ((.22,.19),(.83,1.6)):
        small=dict(a,X=a['X'][:,:8],box=box,l0_penalty=penalty)
        solve(f'public-{i}-oracle-box-{box}',small,reference(small))
    Z=a['X']/np.linalg.norm(a['X'],axis=0)
    x=Z[:,1]; z=Z[:,4]; blend=.55*x+z; blend/=np.linalg.norm(blend)
    y=3.1*x-1.1*blend
    for perturb in (.001,.025):
        X=np.column_stack([x]*9+[blend]*6)+perturb*Z[:,np.arange(15)%Z.shape[1]]
        solve(f'public-{i}-coupled-clusters-{perturb}',dict(a,X=X,y=y,ridge=.015,box=.6,l0_penalty=.15))
