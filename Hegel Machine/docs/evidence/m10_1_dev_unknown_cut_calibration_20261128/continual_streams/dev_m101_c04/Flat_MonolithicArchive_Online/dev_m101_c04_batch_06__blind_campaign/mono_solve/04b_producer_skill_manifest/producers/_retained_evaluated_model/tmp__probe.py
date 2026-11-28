import numpy as np,pickle,glob,importlib.util,itertools
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; n=56
cv=dict(s,train={k:v[:n] if k!='summary' else v for k,v in tr.items()},test={'x':tr['x'][n:]})
def load(f):
 sp=importlib.util.spec_from_file_location('m',f); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
for f in sorted(glob.glob('/tmp/hegel_continual_archive/*_solver.py')):
 try:
  p=load(f).solve_batch(cv); print(f[-12:],*[round(np.mean(abs(p[j]-tr[k][n:])),5) for j,k in [('y_hat','y'),('state_pred','traj')]])
 except Exception as e: print(f[-12:],e)
def aff(x):return np.c_[np.ones(len(x)),x]
def ls(x,y):return np.linalg.lstsq(x,y,rcond=None)[0]
a=s['panel_a']; u,v=a['u'],a['v']; f=aff(u); keep=np.ones(len(u),bool)
for _ in range(4):
 w=ls(f[keep],v[keep]); err=np.linalg.norm(f@w-v,axis=1); keep=err<np.sort(err)[40]+1e-10
print('A affine',w,'errors',np.round(np.sort(err),4)); print('telemetry',s['telemetry_vec']); print('traj sv',np.linalg.svd(tr['traj'],compute_uv=False)); print('traj V',np.linalg.svd(tr['traj'],full_matrices=False)[2][:3]); print('y/state corr',np.corrcoef(np.c_[tr['y'],tr['traj']].T)[0]); print('train summary',tr['summary']);
for name in ('panel_b','panel_c'):
 p=s[name]; u=p['u'];
 for k,v in p.items():
  if k=='u':continue
  for g in [u,np.tanh(u),np.sin(u),np.sign(u)*np.maximum(abs(u)-.35,0)**1.5,np.c_[u,u**2,u**3],np.c_[u,u**2,np.array([u[:,i]*u[:,j] for i,j in itertools.combinations(range(4),2)]).T]]:
   f=aff(g); print(name,k,g.shape[1],round(np.sqrt(np.mean((f@ls(f,v)-v)**2)),6),end='; ')
  print()
