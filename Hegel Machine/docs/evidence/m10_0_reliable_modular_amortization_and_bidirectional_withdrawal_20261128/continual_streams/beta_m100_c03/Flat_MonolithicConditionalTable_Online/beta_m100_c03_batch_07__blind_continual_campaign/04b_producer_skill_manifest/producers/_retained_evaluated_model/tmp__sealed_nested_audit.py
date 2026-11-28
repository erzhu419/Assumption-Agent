import runpy
from pathlib import Path
exec(Path('/tmp/wall_compare.py').read_text().split('for p in itertools.product')[0])
candidate = runpy.run_path('/holdout_task/batch_solver.py')
assert np.allclose(candidate['_calibrate'](s,s['y_obs_raw']), y)
# Add conservative half-cell and face-average interpretations of the wall flux.
cache = {('endpoint',)+p:response(p) for p in itertools.product((0,1),repeat=4)}
for kind in ('face_average','half_cell'):
    mats=[]
    for d,n,h,a in zip('xy',sizes,hs,axes):
        kap=s['kappa_'+d]; b=a.tolil(copy=True)
        for k,l in ((0,1),(n-1,n-2)):
            rate=(kap[k]+kap[l])/(2*h*h) if kind=='face_average' else 2*kap[k]/(h*h)
            b[k,k]=rate; b[k,l]=-rate
        mats.append(b.tocsc())
    a,b=mats; A=(kron(eye(ny),a)+kron(b,eye(nx))+diag).tocsc()
    cache[(kind,)]=np.vstack([splu(A+mu*I).solve(rhs)[inds] for mu in s['mu_shifts']])
base=('endpoint',0,0,0,0); selected=[]; worst={k:0. for k in cache}; worst_nested=0.
for j,test in enumerate(groups):
    train=np.setdiff1d(allrows,test)
    def error(R,tr,te):
        w,g=fit(R,tr)
        return np.sqrt(np.mean((R[te]@w+z[te]@g-y[te])**2))
    inner={k:max(error(R,np.setdiff1d(train,v),v) for l,v in enumerate(groups) if l!=j) for k,R in cache.items()}
    winner=min(inner,key=inner.get); selected.append(winner)
    for k,R in cache.items(): worst[k]=max(worst[k],error(R,train,test))
    worst_nested=max(worst_nested,error(cache[winner],train,test))
print('Nested held-out selections:',selected)
print('Worst nested RMSE:',worst_nested,'fixed sealed:',worst[base])
print('Nearest alternatives:',sorted((err,k) for k,err in worst.items() if k!=base)[:4])
print('Candidate forward matrix vs sealed maximum difference:',np.max(np.abs(cache[base]-candidate['_responses'](s)[0])))
