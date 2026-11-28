import pickle,glob,importlib.util,numpy as np
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x,y,z=(tr[k] for k in ('x','y','traj')); np.set_printoptions(precision=5,suppress=True,linewidth=120)
print('times, telemetry',s['eval_times'],s['telemetry_vec']); print('train ranges',x.min(0),x.max(0)); print('y stats',y.min(),y.max(),y.std()); print('z singular',np.linalg.svd(z,compute_uv=False), 'centered',np.linalg.svd(z-z.mean(0),compute_uv=False)); print('z',z[:8]); print('y',y[:8]);
for k in ('panel_a','panel_b','panel_c'):
 p=s[k]; u=p['u']; a=np.c_[np.ones(len(u)),u]; print(k,'range',u.min(0),u.max(0));
 for h,v in p.items():
  if h=='u':continue
  w=np.linalg.lstsq(a,v,rcond=None)[0]; r=v-a@w; print(h,'affine',w,'resid quantiles',np.quantile(np.linalg.norm(r,axis=1),[0,.5,.8,.9,1])); print('first pairs',np.c_[u,v][:4]);
for f in sorted(glob.glob('/tmp/hegel_continual_archive/*solver.py')):
 l=importlib.util.spec_from_file_location('q',f); m=importlib.util.module_from_spec(l);l.loader.exec_module(m)
 try:
  t={**s,'train':{k:v[:52] for k,v in tr.items()},'test':{'x':x[52:]}}; p=m.solve_batch(t); print(f[-19:],np.mean(abs(p['y_hat']-y[52:])),np.mean(abs(p['state_pred']-z[52:])))
  if hasattr(m,'_inverse'):
   c=m._calibrate(s['panel_a']) if hasattr(m,'_calibrate') else m._calibration(s['panel_a']); print('  cal',c)
 except Exception as e: print(type(e).__name__,e)
