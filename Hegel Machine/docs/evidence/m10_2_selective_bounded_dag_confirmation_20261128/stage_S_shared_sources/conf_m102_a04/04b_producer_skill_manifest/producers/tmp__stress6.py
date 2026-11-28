import pickle, numpy as np, importlib.util
f='/holdout_task/batch_solver.py'; spec=importlib.util.spec_from_file_location('solver',f);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; x=t['x']; y=t['y']; T=t['traj']; n=len(x)
a,b=m._align(s['panel_a']); rad=np.linalg.norm((x-a)/b,axis=1)
models=m.fitc(s['panel_c'],True)
m.fitc=lambda _: models
for name,J in [('public',np.arange(52,72)),('large radius',np.argsort(rad)[-20:]),('small radius',np.argsort(rad)[:20]),('early block',np.arange(20)),('mid block',np.arange(20,40))]+[(f'random {k}',np.random.default_rng(k).permutation(n)[:20]) for k in range(5)]:
 I=np.setdiff1d(np.arange(n),J); z=dict(s,train={k:t[k][I] for k in ['x','y','traj']},test={'x':x[J]}); r=m.solve_batch(z)
 err=abs(r['state_pred']-T[J]); ey=abs(r['y_hat']-y[J]); print(name,'y/traj/terminal/worst horizon',np.round([ey.mean(),err.mean(),err[:,-1].mean(),err.mean(axis=0).max()],6))
