import importlib.util, pickle
from itertools import combinations
import numpy as np
from scipy.optimize import minimize_scalar
spec=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; n=len(tr['y']); rows=np.arange(n)
a,b=m._calibrate(s['panel_a']); z=(tr['x']-a)/b; h=m._operator(s['panel_b'])(z); F=m._quadratic(h)
q=np.column_stack([h]+[(h[:,i]+sign*h[:,j])/np.sqrt(2) for i,j in combinations(range(h.shape[1]),2) for sign in (-1,1)])
q=(q-q[:52].mean(0))/q[:52].std(0)
def fit_rational(train, val, Y):
    def candidate(j,k,idx,ev):
        d=1+k*q[:,j]**2; B=F/d[:,None]; return B[ev]@m._fit(B[idx],Y[idx])
    splits=np.array_split(np.random.default_rng(153).permutation(train),4)
    def loss(j,k):
        return np.mean([np.mean(abs(candidate(j,k,train[~np.isin(train,v)],v)-Y[v])) for v in splits])
    best=(loss(0,0),0,0.)
    for j in range(q.shape[1]):
        r=minimize_scalar(lambda k:loss(j,k),bounds=(0.,1.),method='bounded',options={'xatol':1e-5}); best=min(best,(r.fun,j,r.x))
    return candidate(best[1],best[2],train,val),best[2]
print('public study keys:', sorted(s), 'training keys:', sorted(tr), 'telemetry shape:', np.shape(s['telemetry_vec']))
print('No row-level batch labels, density or value-function fields, or relay modes are supplied; classification and PDE/control tests do not apply.')
for label,idx in [('prescribed',rows), *[(f'tail-{j}',np.argsort(abs(z[:,j]))) for j in range(z.shape[1])]]:
    fit,val=idx[:52],idx[52:]; qraw=np.column_stack([h]+[(h[:,i]+sign*h[:,j])/np.sqrt(2) for i,j in combinations(range(h.shape[1]),2) for sign in (-1,1)]); q=(qraw-qraw[fit].mean(0))/qraw[fit].std(0)
    study=dict(s,train={k:tr[k][fit] for k in ('x','y','traj')},test={'x':tr['x'][val]}); p=m.solve_batch(study); y,T=tr['y'][val],tr['traj'][val]
    yp,k=fit_rational(fit,val,tr['y']); Tp,kt=fit_rational(fit,val,tr['traj'])
    delta=np.max(abs(F@m._fit(F,q*q)-q*q))
    print(label,'baseline scalar/state',np.mean(abs(p['y_hat']-y)),np.mean(abs(p['state_pred']-T)),'rational scalar/state',np.mean(abs(yp-y)),np.mean(abs(Tp-T)),'denominator strengths',k,kt,'pairwise polynomial basis error',delta)
    assert max(np.mean(abs(p['y_hat']-y)),np.mean(abs(p['state_pred']-T)))<.038 and delta<1e-10
