import importlib.util, pickle, numpy as np
p='/holdout_task/'
spec=importlib.util.spec_from_file_location('batch_solver',p+'batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open(p+'public_batch.pkl','rb'))[0]['study']; t=s['train']; n=len(t['x']); order=np.arange(n)
c=m._calibration(s['panel_a']); x=m._invert(t['x'],c); xt=m._invert(s['test']['x'],c)
print('calibration',c,'abs max train/test',abs(x).max(),abs(xt).max())
for z,draw in [('prescribed',order), *[(f'random{i}',np.random.default_rng(i).permutation(n)) for i in range(8)], ('largest radius',np.argsort(np.linalg.norm(x,axis=1))),('largest absolute input',np.argsort(np.max(abs(x),axis=1))),('largest scalar',np.argsort(abs(t['y']))),('largest final state',np.argsort(abs(t['traj'][:,-1]))), *[(f'coordinate{j}/{sg}',np.argsort(sg*x[:,j])) for j in range(4) for sg in (-1,1)]]:
    i,j=draw[:52],draw[52:]; sub=dict(s,train={k:v[i] if k!='summary' else v for k,v in t.items()},test={'x':t['x'][j]}); o=m.solve_batch(sub)
    e=abs(o['state_pred']-t['traj'][j]); a=abs(o['y_hat']-t['y'][j]); print(z,*np.round([a.mean(),e.mean(),np.max(e.mean(axis=0)),a.max(),e.max()],6)); assert max(a.mean(),e.mean(),np.max(e.mean(axis=0))) < .038
out=m.solve_batch(s); assert all(np.all(np.isfinite(v)) for v in out.values()); print('clean-process outputs',[(k,v.shape) for k,v in out.items()])
