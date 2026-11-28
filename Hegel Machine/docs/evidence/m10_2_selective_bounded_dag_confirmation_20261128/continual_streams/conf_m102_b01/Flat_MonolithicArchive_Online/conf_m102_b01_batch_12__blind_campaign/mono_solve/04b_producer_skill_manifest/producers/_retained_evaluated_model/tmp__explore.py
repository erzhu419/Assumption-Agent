import numpy as np,pickle,glob,importlib.util,copy
np.set_printoptions(precision=5,suppress=True,linewidth=120)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; print('times',s['eval_times'],'telemetry',s['telemetry_vec'],'summary',tr['summary']);
for k in ['x','y','traj']:
 a=tr[k]; print(k,'min',a.min(0),'max',a.max(0),'std',a.std(0),'first',a[:5]);
print('traj sv',np.linalg.svd(tr['traj'],compute_uv=False))
for k in 'abc':
 a=s['panel_'+k]; u=a['u']; print(k,'u',u[:3],u.min(0),u.max(0));
 def aff(x):return np.c_[np.ones(len(x)),x]
 for name,v in a.items():
  if name=='u':continue
  print(name,'first',v[:2],'sv',np.linalg.svd(v-v.mean(0),compute_uv=False))
  for desc,d in [('linear',aff(u)),('dead',aff(np.sign(u)*np.maximum(abs(u)-.35,0)**1.5)),('tanh',aff(np.tanh(1.8*u))),('pairs',aff(np.c_[*[u[:,i]*u[:,j] for i in range(4) for j in range(i+1,4)]])),('poly2',aff(np.c_[u,*[u[:,i]*u[:,j] for i in range(4) for j in range(i,4)]]))]:
   m=np.arange(len(v))
   for _ in range(6):
    r=v-d@np.linalg.lstsq(d[m],v[m],rcond=None)[0]; m=np.argsort(np.linalg.norm(r,axis=1))[:len(v)-max(3,len(v)//8)]
   print(desc,np.median(abs(r)),np.mean(abs(r)),end='; ')
  print()
cv=copy.copy(s); cv['train']={k:(v[:52] if k in ('x','y','traj') else v) for k,v in tr.items()}; cv['test']={'x':tr['x'][52:]}
for f in glob.glob('/tmp/hegel_continual_archive/*solver.py'):
 try:
  spec=importlib.util.spec_from_file_location('solver',f);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m); p=m.solve_batch(cv); print(f.split('/')[-1],np.mean(abs(p['y_hat']-tr['y'][52:])),np.mean(abs(p['state_pred']-tr['traj'][52:])))
 except Exception as e:print(f,e)
