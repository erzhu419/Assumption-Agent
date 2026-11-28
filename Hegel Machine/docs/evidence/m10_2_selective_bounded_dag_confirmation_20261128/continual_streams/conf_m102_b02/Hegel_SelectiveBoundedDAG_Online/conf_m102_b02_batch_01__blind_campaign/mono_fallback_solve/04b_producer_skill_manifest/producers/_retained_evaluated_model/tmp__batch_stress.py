import importlib.util, pickle, numpy as np
path='/holdout_task/batch_solver.py'
spec=importlib.util.spec_from_file_location('batch_solver',path)
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']
x=m._calibrate(s['panel_a'],tr['x']); rng=np.random.default_rng(417)
def score(p):
    i,j=p[:52],p[52:]
    out=m.solve_batch(dict(s,train=dict(tr,x=tr['x'][i],y=tr['y'][i],traj=tr['traj'][i]),test={'x':tr['x'][j]}))
    err=np.abs(out['state_pred']-tr['traj'][j]); return np.r_[np.mean(abs(out['y_hat']-tr['y'][j])),err.mean(),err.mean(0)]
a=score(np.arange(72)); r=np.stack([score(rng.permutation(72)) for _ in range(30)]); b=np.stack([score(np.argsort(sign*x[:,k])) for k in range(4) for sign in [-1,1]])
for k,v in [('fixed 52/20',a),('worst random 52/20',r.max(0)),('worst coordinate-tail 52/20',b.max(0))]: print(k,'y, trajectory, and eight horizons:',v.round(6))
assert max(a[:2])<=.038
out=m.solve_batch(s); y=out['y_hat']; z=out['state_pred']; assert y.shape==(36,) and z.shape==(36,8) and out['diagnostics_vec'].shape==(4,)
assert all(np.isfinite(v).all() for v in out.values()); np.testing.assert_allclose(out['diagnostics_vec'],[y.mean(),y.std(),z[:,-1].mean(),z[:,-1].std()],rtol=0,atol=0)
print('isolated artifact checks passed')
