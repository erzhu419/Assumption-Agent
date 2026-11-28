import importlib.util
import pickle
import numpy as np
spec=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl','rb') as f:s=pickle.load(f)[0]['study']
x=s['train']['x']; y=s['train']['y']; z=s['train']['traj']; rng=np.random.default_rng(91)
cv=[(np.arange(56),np.arange(56,72))]+[(p[:56],p[56:]) for p in (rng.permutation(72) for _ in range(7))]
r=np.linalg.norm(x,axis=1); p=np.argsort(r); stress=[(p[:56],p[56:])]
base,d=m._kernel_inputs(s,x); a=base+.3*np.exp(-.5*d); b=base+np.exp(-.125*d)
rows=[]
for i,j in cv+stress:
 ss={**s,'train':{**s['train'],'x':x[i],'y':y[i],'traj':z[i]},'test':{'x':x[j]}}
 o=m.solve_batch(ss); p,q=o['y_hat'],o['state_pred']; end=q[:,-1]
 assert np.allclose(o['diagnostics_vec'],[p.mean(),p.std(),end.mean(),end.std()])
 v=np.linalg.svd(z[i],full_matrices=False)[2][0]; projected=(q@v)[:,None]*v
 rows.append([abs(p-y[j]).mean(),abs(q-z[j]).mean(),abs(projected-z[j]).mean(),abs(end-z[j,-1]).mean(),abs(q-z[j]).mean(axis=0).max()])
rows=np.array(rows);print('y, trajectory, rank-one trajectory, final horizon, worst horizon'); print('56/16',rows[0]);print('8-fold mean',rows[:-1].mean(axis=0));print('8-fold worst',rows[:-1].max(axis=0));print('outer-radius split',rows[-1]);
o=m.solve_batch(s);assert o['y_hat'].shape==(36,) and o['state_pred'].shape==(36,8) and o['diagnostics_vec'].shape==(4,) and all(np.all(np.isfinite(a)) for a in o.values())
