import pickle,time,itertools
import numpy as np
from scipy.optimize import lsq_linear
from certified_box_l0 import solve_certified_box_l0

def check(q, oracle=False):
    X,y=q['X'],q['y']; p=X.shape[1]; box=q['box']; ridge=q['ridge']; pen=q['l0_penalty']
    t=time.perf_counter(); r=solve_certified_box_l0(q); sec=time.perf_counter()-t
    w=np.array(r['weights']); rr=X@w-y
    value=(rr@rr+ridge*(w@w))/2+pen*np.count_nonzero(abs(w)>1e-10)
    lb=r['certified_lower_bound']; leaves=r['certificate_leaves']
    assert w.shape==(p,) and np.max(abs(w),initial=0)<=box and abs(value-r['objective'])<=1e-7
    assert 0<=r['objective']-lb<=q['gap_tol'],r
    assert 1<=len(leaves)<=q['max_leaves'] and lb<=min(l['leaf_lower_bound'] for l in leaves)+1e-8
    cover=np.zeros(1<<p,dtype=int); all_masks=np.arange(1<<p)
    for l in leaves:
        z,o=l['fixed_zero'],l['fixed_nonzero']; assert len(set(z+o))==len(z+o) and all(0<=j<p for j in z+o)
        zm,om=sum(1<<j for j in z),sum(1<<j for j in o)
        cover[((all_masks&zm)==0)&((all_masks&om)==om)]+=1
    assert np.all(cover==1)
    if oracle:
        A=np.vstack([X,np.sqrt(ridge)*np.eye(p)]); b=np.r_[y,np.zeros(p)]
        direct=[]
        for mask in all_masks:
            idx=np.flatnonzero(mask & (1<<np.arange(p)))
            z=lsq_linear(A[:,idx],b,bounds=(-box,box),method='trf',tol=1e-13).x if idx.size else np.zeros(0)
            res=A[:,idx]@z-b; direct.append((res@res)/2+pen*idx.size)
        assert abs(min(direct)-value)<=1e-7 and lb<=min(direct)+1e-9,(min(direct),r)
    print(p,'box',box,'ridge',ridge,'time',round(sec,3),'gap',r['objective']-lb,flush=True)

cases=pickle.load(open('/holdout_task/public_box_l0_cases.pkl','rb'))['cases']
for q in cases: check(q)
rng=np.random.default_rng(682)
B=rng.normal(size=(40,5)); X=B[:,rng.integers(0,5,size=15)]+rng.normal(size=(40,15))*.03
for dim,box,ridge in [(7,.2,0),(7,2.,0),(15,.19,0),(15,.5,.04),(15,1.2,.04)]:
    xx=X[:,:dim].copy(); xx[:,-1]=xx[:,0]; y=2*B[:,0]-1.4*B[:,1]+.7*B[:,3]+.14*rng.normal(size=40)
    check(dict(X=xx,y=y,box=box,ridge=ridge,l0_penalty=.28,gap_tol=1e-6,max_leaves=512),oracle=dim<10)
