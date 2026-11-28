import importlib.util,glob,pickle,numpy as np,traceback
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; a=s['train']; p=dict(s,train={k:v[:52] if k!='summary' else v for k,v in a.items()},test={'x':a['x'][52:]})
for fn in sorted(glob.glob('/tmp/hegel_continual_archive/*solver.py')):
 try:
  m=importlib.util.module_from_spec(q:=importlib.util.spec_from_file_location('m',fn));q.loader.exec_module(m); o=m.solve_batch(p);print(fn.split('_')[-2],np.mean(abs(o['y_hat']-a['y'][52:])),np.mean(abs(o['state_pred']-a['traj'][52:])))
 except Exception as e:print(fn,type(e).__name__,str(e))
