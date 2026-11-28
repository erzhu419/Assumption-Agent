import glob, importlib.util, pickle, numpy as np
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; k=52
q=dict(s,train=dict(t,**{z:t[z][:k] for z in ('x','y','traj')}),test={'x':t['x'][k:]}); q['train']['summary']=np.array([t['y'][:k].mean(),t['y'][:k].std(),t['traj'][:k,-1].mean(),t['traj'][:k,-1].std()]); print('telemetry',s['telemetry_vec'])
for f in sorted(glob.glob('/tmp/hegel_continual_archive/*solver.py')):
 try:
  m=importlib.util.module_from_spec(sp:=importlib.util.spec_from_file_location('p',f)); sp.loader.exec_module(m); r=m.solve_batch(q); print(f.split('/')[-1],np.mean(abs(r['y_hat']-t['y'][k:])),np.mean(abs(r['state_pred']-t['traj'][k:]),axis=0))
 except Exception as e: print(f,repr(e))
for name in ('panel_a','panel_b','panel_c'):
 d=s[name]; u=d['u']; z=np.c_[np.ones(len(u)),u]; print(name)
 for key in d.keys()-{'u'}:
  v=d[key]; c=np.linalg.lstsq(z,v,rcond=None)[0]; print(key,'linear coefficients',np.round(c,3),'abs residual quantiles',np.quantile(abs(v-z@c),[.25,.5,.9,1]))
print('trajectory svd',np.linalg.svd(t['traj'],compute_uv=False))
