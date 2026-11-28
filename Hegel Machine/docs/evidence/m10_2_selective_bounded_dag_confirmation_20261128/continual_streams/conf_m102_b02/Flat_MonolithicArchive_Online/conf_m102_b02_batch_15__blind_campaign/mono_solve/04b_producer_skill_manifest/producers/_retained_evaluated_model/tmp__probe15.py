import pickle,numpy as np,glob,importlib.util,copy
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; np.set_printoptions(precision=5,suppress=True,linewidth=130)
for k,p in s.items():
 if isinstance(p,dict):
  for n,a in p.items(): print(k,n,'mean',np.mean(a,axis=0),'std',np.std(a,axis=0),'head',a[:3]);
 else: print(k,p)
for k in ('panel_a','panel_b','panel_c'):
 p=s[k]; u=p['u']; A=np.c_[np.ones(len(u)),u]
 for n,v in p.items():
  if n=='u':continue
  m=np.linalg.lstsq(A,v,rcond=None)[0]; e=np.linalg.norm(A@m-v,axis=1); print(k,n,m,'resid',np.sort(e))
t=copy.deepcopy(s);t['test']['x']=tr['x'][52:];t['train']={k:(v[:52] if len(v)==72 else v) for k,v in tr.items()}
for f in sorted(glob.glob('/tmp/hegel_continual_archive/*solver.py')):
 spec=importlib.util.spec_from_file_location('old',f); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 try:
  r=m.solve_batch(t);print(f.split('/')[-1],np.mean(abs(r['y_hat']-tr['y'][52:])),np.mean(abs(r['state_pred']-tr['traj'][52:])),np.mean(abs(r['state_pred']-tr['traj'][52:]),axis=0))
 except Exception as e: print(type(e).__name__,str(e))
print('traj svd',np.linalg.svd(tr['traj'],compute_uv=False)); print('corr Y T',np.corrcoef(np.c_[tr['y'],tr['traj']].T)[0]);print('trajectories',tr['traj'][:12]);
