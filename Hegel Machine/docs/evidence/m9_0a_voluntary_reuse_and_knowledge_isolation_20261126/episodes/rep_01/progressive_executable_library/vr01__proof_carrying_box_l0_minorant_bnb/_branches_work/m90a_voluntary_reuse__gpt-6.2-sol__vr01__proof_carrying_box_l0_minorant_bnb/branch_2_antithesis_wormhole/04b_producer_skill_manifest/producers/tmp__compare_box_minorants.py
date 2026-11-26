import pickle, runpy, time
import numpy as np
v=runpy.run_path('/tmp/verify_proof_box.py')
s=v['s']; verify=v['verify']; old=s._diagonal

def candidate(G):
    d=old(G)
    unseen=set(range(len(G)))
    while unseen:
        block=[unseen.pop()]
        for j in block:
            near=[k for k in unseen if G[j,k]!=0]
            block.extend(near); unseen.difference_update(near)
        B=G[np.ix_(block,block)]
        if np.any(B.diagonal()<=0):
            continue
        scale=np.sqrt(B.diagonal())
        R=B/scale[:,None]/scale[None,:]
        margin=64*np.finfo(float).eps*len(B)*max(1.,np.max(np.abs(R)))
        floor=np.linalg.eigvalsh(R)[0]-margin
        if floor>np.mean(d[block]/B.diagonal()):
            d[block]=floor*B.diagonal()
    return d

for i,c in enumerate(pickle.load(open('/holdout_task/public_proof_box_l0_cases.pkl','rb'))):
    p=c['problem']; X=p['X']; rng=np.random.default_rng(901+i); order=rng.permutation(X.shape[1])
    for span in (1.,4.,16.,64.,256.):
        scales=np.empty(X.shape[1]); scales[order]=np.geomspace(span**-.5,span**.5,X.shape[1]); q=dict(p,X=X*scales)
        outcomes=[]
        for method in (old,candidate):
            s._diagonal=method; start=time.perf_counter()
            try:
                r=s.solve_proof_carrying_box_l0(q)
                leaves,gap,eig,_=verify(q,r,c['gt']['exact_objective'] if span==1 else None)
                outcomes.append((leaves,gap,eig,round(time.perf_counter()-start,3)))
            except (AssertionError,RuntimeError) as e:
                outcomes.append(repr(e))
        print(i,span,'baseline / normalized',outcomes,flush=True)
