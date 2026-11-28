import importlib.util
import pickle
import numpy as np
from scipy.optimize import least_squares
p='/holdout_task/'
spec=importlib.util.spec_from_file_location('batch_solver',p+'batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
with open(p+'public_batch.pkl','rb') as f: s=pickle.load(f)[0]['study']
tr=s['train']; x=m._coordinates(s,tr['x']); y=tr['y']; z=tr['traj']; t=s['eval_times']; ids=np.arange(len(x)); d=m._scalar_design(x)
aff=lambda q:np.c_[np.ones(len(q)),q]
controls={'intercept+linear':aff(x),'tied':d,'difference':np.c_[x,x[:,0]*x[:,2]-x[:,1]*x[:,3]],'raw tied':np.c_[aff(tr['x']),tr['x'][:,0]*tr['x'][:,2]+tr['x'][:,1]*tr['x'][:,3]]}
def joint(it,va):
    q=np.linalg.qr(x[it],mode='reduced')[0]; target=(q.T@z[it]).T
    def basis(r): return -np.expm1(-t[:,None]*r)
    def resid(r):
        b=basis(r); return (b@m._fit(b,target)-target).ravel()
    rates=least_squares(resid,[.4,1.8],bounds=([.01,.8],[.8,8.])).x
    b=basis(rates); return x[va]@m._fit(x[it],m._fit(b,z[it].T).T)@b.T
families={'last16':[ids[56:]],'blocked':[ids[i:i+16] for i in range(0,len(ids),16)],'random':np.array_split(np.random.default_rng(7).permutation(ids),5),'radius':[np.argsort(np.linalg.norm(x,axis=1))[-16:]],'tails':[np.argsort(x[:,j])[sl] for j in range(4) for sl in (slice(0,12),slice(-12,None))]}
for family,folds in families.items():
    errors=[]; coeff=[]; state=[]
    for va in folds:
        it=np.setdiff1d(ids,va)
        h=dict(s,train={k:tr[k][it] for k in ('x','y','traj')},test={'x':tr['x'][va]}); pred=m.solve_batch(h)
        np.testing.assert_allclose(pred['y_hat'],d[va]@m._fit(d[it],y[it]),rtol=1e-12,atol=1e-12)
        errors.append([np.mean(abs(y[va]-v[va]@m._fit(v[it],y[it]))) for v in controls.values()]); coeff.append(m._fit(d[it],y[it])[-1])
        state.append([np.mean(abs(z[va]-pred['state_pred']),axis=0),np.mean(abs(z[va]-joint(it,va)),axis=0)])
    print(family,list(controls),'scalar mean',np.mean(errors,axis=0),'worst',np.max(errors,axis=0),'interaction coefficient range',(min(coeff),max(coeff)))
    print(family,'state [shipped,joint] mean',np.mean(state,axis=(0,2)),'worst horizon',np.max(state,axis=(0,2)))
