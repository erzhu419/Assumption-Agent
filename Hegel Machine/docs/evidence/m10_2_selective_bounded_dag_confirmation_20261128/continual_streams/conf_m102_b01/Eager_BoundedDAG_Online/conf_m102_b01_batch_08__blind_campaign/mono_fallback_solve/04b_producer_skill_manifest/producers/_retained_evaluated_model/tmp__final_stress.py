import importlib.util,pickle,numpy as np
p='/holdout_task/batch_solver.py'; sp=importlib.util.spec_from_file_location('solver',p); m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; cases=[(np.arange(52),np.arange(52,72)),(np.arange(20,72),np.arange(20))]; mets=[]
for model in [m._ensemble_baseline,m._updated]:
 scores=[]
 for a,b in cases:
  q=dict(s,train={k:tr[k][a] for k in ('x','y','traj')},test={'x':tr['x'][b]}); p=model(q); dy=abs(p['y_hat']-tr['y'][b]);dt=abs(p['state_pred']-tr['traj'][b]); assert np.isfinite(dy).all() and np.isfinite(dt).all(); scores.append([dy.mean(),dt.mean()]);print(model.__name__,np.round(scores[-1],6),'worst horizon',round(dt.mean(0).max(),6),flush=True)
 mets.append(np.array(scores))
# Keep the previous method for any head not improved in aggregate and at the specified gate.
old,new=mets
keep=(new[0] <= old[0]) & ((new.mean(0) <= old.mean(0)) | (new[0] <= .038))
if not keep.any():
 import shutil; shutil.copyfile('/tmp/kernel_baseline.py','/holdout_task/batch_solver.py')
elif not keep.all():
 with open('/holdout_task/batch_solver.py','a') as f:
  f.write('\n\ndef solve_batch(study):\n    old=_ensemble_baseline(study)\n    new=_updated(study)\n    y='+('new' if keep[0] else 'old')+"['y_hat']\n    t="+('new' if keep[1] else 'old')+"['state_pred']\n    return {'y_hat':y,'state_pred':t,'diagnostics_vec':np.array([y.mean(),y.std(),t[:,-1].mean(),t[:,-1].std()])}\n")
print('retained adaptive heads:',keep.tolist(),flush=True)
