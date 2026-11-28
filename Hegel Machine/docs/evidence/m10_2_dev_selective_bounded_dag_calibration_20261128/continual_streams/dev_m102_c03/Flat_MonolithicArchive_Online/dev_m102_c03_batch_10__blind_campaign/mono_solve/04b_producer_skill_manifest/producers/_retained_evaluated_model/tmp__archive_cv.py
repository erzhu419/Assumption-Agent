import glob,importlib.util,copy,pickle,numpy as np
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=copy.deepcopy(s); c['test']['x']=s['train']['x'][52:]; c['train']={k:v[:52] if k!='summary' else v for k,v in s['train'].items()}
for f in sorted(glob.glob('/tmp/hegel_continual_archive/*solver.py')):
 try:
  spec=importlib.util.spec_from_file_location('candidate',f); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); out=m.solve_batch(c); print(f.split('/')[-1],np.mean(abs(out['y_hat']-s['train']['y'][52:])),np.mean(abs(out['state_pred']-s['train']['traj'][52:])))
 except Exception as e: print(f,str(e))
