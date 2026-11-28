import glob, importlib.util, pickle, numpy as np
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; q={**s,'train':{**tr,**{k:tr[k][:52] for k in ['x','y','traj']}},'test':{'x':tr['x'][52:]}}; q['train']['summary']=np.array([tr['y'][:52].mean(),tr['y'][:52].std(),tr['traj'][:52,-1].mean(),tr['traj'][:52,-1].std()]);
for f in sorted(glob.glob('/tmp/hegel_continual_archive/*solver.py')):
 try:
  p=importlib.util.spec_from_file_location('candidate',f); m=importlib.util.module_from_spec(p); p.loader.exec_module(m); r=m.solve_batch(q); print(f.split('/')[-1],np.abs(r['y_hat']-tr['y'][52:]).mean(),np.abs(r['state_pred']-tr['traj'][52:]).mean(),flush=True)
 except Exception as e: print(f,type(e).__name__,str(e),flush=True)
