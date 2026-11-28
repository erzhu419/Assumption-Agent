import importlib.util,pickle,numpy as np
path='/holdout_task/batch_solver.py'
spec=importlib.util.spec_from_file_location('verified_solver',path); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; n=len(tr['y']); ar=np.arange(n); rng=np.random.default_rng(34)
folds=[(ar[:52],ar[52:])]+[(i[:52],i[52:]) for i in [rng.permutation(n) for _ in range(12)]]+[(np.r_[ar[:j],ar[j+20:]],ar[j:j+20]) for j in [0,13,26,39]]
def check(out,n):
 y,t,d=[out[k] for k in ('y_hat','state_pred','diagnostics_vec')]; assert y.shape==(n,) and t.shape==(n,len(s['eval_times'])) and d.shape==(4,); assert all(np.isfinite(a).all() for a in (y,t,d)); np.testing.assert_allclose(d,[y.mean(),y.std(),t[:,-1].mean(),t[:,-1].std()]); return y,t
scores=[]; horizons=[]
for t,v in folds:
 f=dict(s,train=dict(tr,x=tr['x'][t],y=tr['y'][t],traj=tr['traj'][t]),test=dict(x=tr['x'][v])); y,a=check(m.solve_batch(f),len(v)); e=abs(a-tr['traj'][v]); scores.append((abs(y-tr['y'][v]).mean(),e.mean())); horizons.append(e.mean(axis=0))
scores=np.array(scores); print('fixed / mean / worst y and trajectory MAE:',scores[0],scores.mean(0),scores.max(0)); print('worst horizon MAE:',np.max(horizons)); assert np.max(scores[0])<=.038 and max(scores[:,0])<.055; check(m.solve_batch(s),len(s['test']['x']))
pa=s['panel_a']; e=np.linalg.norm(m._decode(pa['v'],m._calibration(pa))-pa['u'],axis=1); print('calibration inverse row error median, p90:',np.quantile(e,[.5,.9]))
