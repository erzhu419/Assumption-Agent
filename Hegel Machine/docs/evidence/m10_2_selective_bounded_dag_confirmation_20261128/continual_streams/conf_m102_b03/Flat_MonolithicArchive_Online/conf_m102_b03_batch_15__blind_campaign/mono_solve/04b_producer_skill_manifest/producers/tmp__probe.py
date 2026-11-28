import pickle,numpy as np,glob,importlib.util
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; T=tr['traj']; print('times/telemetry',s['eval_times'], s['telemetry_vec'],'trajectory sv',np.linalg.svd(T,compute_uv=False))
c={**s,'train':{k:v[:52] if k!='summary' else np.array([tr['y'][:52].mean(),tr['y'][:52].std(),T[:52,-1].mean(),T[:52,-1].std()]) for k,v in tr.items()},'test':{'x':tr['x'][52:]}}
for f in sorted(glob.glob('/tmp/hegel_continual_archive/*solver.py')):
 sp=importlib.util.spec_from_file_location('m',f);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
 try:
  p=m.solve_batch(c); print(f.split('_')[-2],*[np.mean(np.abs(p[k]-a[52:])) for k,a in [('y_hat',tr['y']),('state_pred',T)]])
 except Exception as e:print(f,type(e).__name__,e)
for key in ['panel_a','panel_b','panel_c']:
 p=s[key]; u=p['u']; print(key)
 for k,v in p.items():
  if k=='u':continue
  for a in [np.c_[np.ones(len(u)),u],np.c_[np.ones(len(u)),u,u*u],np.c_[np.ones(len(u)),u,np.tanh(1.8*u),np.sin(2.4*u)]]:
   e=np.abs(a@np.linalg.lstsq(a,v,rcond=None)[0]-v);print(k,a.shape[1],np.quantile(e,[.1,.5,.9,1]))
