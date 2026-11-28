import glob,importlib.util,copy,pickle,numpy as np
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; q=copy.deepcopy(s); q['train']={k:v[:52] if k!='summary' else np.array([t['y'][:52].mean(),t['y'][:52].std(),t['traj'][:52,-1].mean(),t['traj'][:52,-1].std()]) for k,v in t.items()}; q['test']={'x':t['x'][52:]}
for k,v in s.items():
 print(k, {a:(b.shape,np.round(b.reshape(-1)[:8],4)) for a,b in v.items()} if isinstance(v,dict) else v)
for f in sorted(glob.glob('/tmp/hegel_continual_archive/*solver.py')):
 try:
  sp=importlib.util.spec_from_file_location('candidate',f); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m); o=m.solve_batch(copy.deepcopy(q)); print(f.split('/')[-1],*[np.mean(abs(o[a]-t[b][52:])) for a,b in [('y_hat','y'),('state_pred','traj')]])
 except Exception as e: print(f,e)
for k in ['panel_a','panel_b','panel_c']:
 p=s[k]; u=p['u']; A=np.c_[np.ones(len(u)),u];
 for key,v in p.items():
  if key=='u':continue
  residual=v-A@np.linalg.lstsq(A,v,rcond=None)[0]; print(k,key,'OLS RMS',np.sqrt(np.mean(residual**2)), 'row errors',np.round(np.linalg.norm(residual,axis=1),3))
print('trajectory sv',np.linalg.svd(t['traj'],compute_uv=False)); print('trajectory sample',np.round(t['traj'][:5],5))
