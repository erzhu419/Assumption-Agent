import numpy as np, pickle, pathlib, importlib.util
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x,y,T=(tr[k] for k in ('x','y','traj'))
ls=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]
aff=lambda x:np.c_[np.ones(len(x)),x]
np.set_printoptions(precision=5,suppress=True)
print('telemetry',s['telemetry_vec'],'trajectory singular values',np.linalg.svd(T,compute_uv=False))
for p in pathlib.Path('/tmp/hegel_continual_archive').glob('*solver.py'):
 sp=importlib.util.spec_from_file_location('prior',p); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
 ss=dict(s,train={k:v[:56] for k,v in tr.items()},test={'x':x[56:]})
 try:
  o=m.solve_batch(ss); print(p.name, np.mean(abs(y[56:]-o['y_hat'])),np.mean(abs(T[56:]-o['state_pred'])))
 except Exception as e: print(p.name,e)
for p in ('panel_a','panel_b','panel_c'):
 u=s[p]['u']; keep=np.max(abs(s[p].get('v',u)-u),axis=1)<2 if p=='panel_a' else np.ones(len(u),bool); u=u[keep]
 for name,V in s[p].items():
  if name=='u':continue
  v=V[keep]; print(p,name,len(u));
  for a in [aff(u),np.c_[aff(u),u**2,u**3],np.c_[aff(u),u**3,((u-1)*np.roll(u,-1,axis=1))]]:
   r=v-a@ls(a,v); print('rmse / max',np.sqrt(np.mean(r*r)),np.max(abs(r)))
