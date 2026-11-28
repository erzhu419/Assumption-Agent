import pickle,numpy as np, importlib.util
sp=importlib.util.spec_from_file_location('batch_solver','/holdout_task/batch_solver.py'); mod=importlib.util.module_from_spec(sp);sp.loader.exec_module(mod)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; n=len(t['x']); norm=np.linalg.norm(mod._calibration(s['panel_a'], t['x']),axis=1); a=np.arange(n)
d={ '52/20':a[52:], 'large-coordinate':np.argsort(norm)[-20:], 'small-coordinate':np.argsort(norm)[:20], 'large-readout':np.argsort(abs(t['y']))[-20:], 'late-amplitude':np.argsort(abs(t['traj'][:,-1]))[-20:]};r=np.random.default_rng(220)
d.update({f'random_{i}':r.permutation(n)[:20] for i in range(12)})
worst=np.zeros(4)
for name,j in d.items():
 i=np.setdiff1d(a,j); fit={k: (v[i] if k in ('x','y','traj') else v) for k,v in t.items()}; p=mod.solve_batch(dict(s,train=fit,test={'x':t['x'][j]})); es=abs(p['state_pred']-t['traj'][j]); ey=abs(p['y_hat']-t['y'][j]); e=np.array([ey.mean(),es.mean(),es[:,-1].mean(),es.mean(0).max()]); print(name,e);worst=np.maximum(worst,e)
print('Worst (readout, trajectory, terminal, worst time):',worst)
