import glob,importlib.util,pickle,copy,time
import numpy as np
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; p=copy.deepcopy(s); p['train']={k:(v[:52] if k!='summary' else np.array([t['y'][:52].mean(),t['y'][:52].std(),t['traj'][:52,-1].mean(),t['traj'][:52,-1].std()])) for k,v in t.items()}; p['test']={'x':t['x'][52:]}
for f in sorted(glob.glob('/tmp/hegel_continual_archive/*solver.py')):
 try:
  spec=importlib.util.spec_from_file_location('m',f); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); now=time.monotonic(); r=m.solve_batch(p); print(f.split('/')[-1], 'MAEs', np.abs(r['y_hat']-t['y'][52:]).mean(),np.abs(r['state_pred']-t['traj'][52:]).mean(),'seconds',time.monotonic()-now)
 except Exception as e: print(f,repr(e))
