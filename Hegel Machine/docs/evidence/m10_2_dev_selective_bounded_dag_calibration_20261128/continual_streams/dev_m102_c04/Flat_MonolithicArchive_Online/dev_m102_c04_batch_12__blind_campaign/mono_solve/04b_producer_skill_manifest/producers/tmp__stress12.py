import sys, pickle, importlib.util
import numpy as np
path='/holdout_task/batch_solver.py'
spec=importlib.util.spec_from_file_location('batch_solver',path)
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
t=s['train']; x=t['x']; idx=np.arange(len(x)); rng=np.random.default_rng(1207)
def run(held):
    fit=np.setdiff1d(idx,held)
    st=dict(s, train={k:t[k][fit] for k in ('x','y','traj')}, test={'x':x[held]})
    out=mod.solve_batch(st)
    ey=abs(out['y_hat']-t['y'][held]); es=abs(out['state_pred']-t['traj'][held]); final=out['state_pred'][:,-1]
    np.testing.assert_array_equal(out['diagnostics_vec'], [out['y_hat'].mean(),out['y_hat'].std(),final.mean(),final.std()])
    return np.array([ey.mean(), es.mean(), es.mean(0).max(),es[:,-1].mean()])
print('columns: scalar MAE, trajectory MAE, worst-horizon MAE, terminal MAE')
public=run(idx[52:]); print('required 52/20:',public)
assert max(public)<.038
random=np.array([run(rng.choice(idx,20,replace=False)) for _ in range(30)])
print('30 randomized 52/20 splits, worst:',random.max(0))
z=mod._calibrate(s['panel_a'],x); p,r=mod._operators(s['panel_c'],z)
orders=np.c_[z,np.linalg.norm(z,axis=1),np.sum(p*r,axis=1),p[:,0]*r[:,1]]
slices=np.array([run(np.argsort(col)[sl]) for col in orders.T for sl in (slice(None,20),slice(-20,None))])
print('14 one-sided operator/input slices, worst:',slices.max(0))
assert max(random.max(0).max(),slices.max(0).max()) < .038
out=mod.solve_batch(s)
assert out['y_hat'].shape==(36,) and out['state_pred'].shape==(36,8)
assert all(np.isfinite(z).all() for z in out.values())
print('clean isolated import and full-study inference passed')
