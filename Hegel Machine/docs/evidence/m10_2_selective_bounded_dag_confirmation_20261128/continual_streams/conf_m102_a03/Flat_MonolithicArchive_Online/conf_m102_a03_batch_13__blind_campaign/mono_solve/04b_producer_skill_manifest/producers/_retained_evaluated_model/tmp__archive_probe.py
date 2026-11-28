import glob,importlib.util,pickle,numpy as np
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; ss={**s,'train':{k:(v[:52] if k!='summary' else np.array([tr['y'][:52].mean(),tr['y'][:52].std(),tr['traj'][:52,-1].mean(),tr['traj'][:52,-1].std()])) for k,v in tr.items()},'test':{'x':tr['x'][52:]}}
print('telemetry',s['telemetry_vec'],'traj sv',np.linalg.svd(tr['traj'],compute_uv=False),'target stats',np.std(tr['y']),np.std(tr['traj'],axis=0))
for f in sorted(glob.glob('/tmp/hegel_continual_archive/*solver.py')):
 try:
  sp=importlib.util.spec_from_file_location('m',f); m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);o=m.solve_batch(ss);print(f.split('/')[-1],np.mean(abs(o['y_hat']-tr['y'][52:])),np.mean(abs(o['state_pred']-tr['traj'][52:])))
 except Exception as e: print(f,e)
