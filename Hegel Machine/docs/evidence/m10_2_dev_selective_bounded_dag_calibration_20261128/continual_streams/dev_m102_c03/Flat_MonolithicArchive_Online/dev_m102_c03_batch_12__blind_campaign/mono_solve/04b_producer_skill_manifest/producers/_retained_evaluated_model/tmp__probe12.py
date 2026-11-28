import numpy as np, pickle, glob, importlib.util
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; cv=dict(s,train={k:v[:52] if k!='summary' else v for k,v in t.items()},test={'x':t['x'][52:]})
for p in glob.glob('/tmp/hegel_continual_archive/*solver.py'):
 try:
  sp=importlib.util.spec_from_file_location('m',p); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m); o=m.solve_batch(cv); print(p.split('/')[-1], *(np.mean(np.abs(o[k]-t[j][52:])) for k,j in [('y_hat','y'),('state_pred','traj')]))
 except Exception as e: print(p,repr(e))
