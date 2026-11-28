import numpy as np, pickle, glob, importlib.util
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; t=s['eval_times']; T=tr['traj']
def D(x):return np.c_[np.ones(len(x)),x]
def ls(a,b):return np.linalg.lstsq(a,b,rcond=1e-10)[0]
print('T singular values',np.linalg.svd(T,compute_uv=False)); print('T quantiles',np.quantile(T,[0,.5,1],axis=0))
for k in range(1,4):
 def res(p):
  H=-np.expm1(-t[:,None]*p);return (H@ls(H,T.T)-T.T).ravel()
 z=least_squares(res,np.geomspace(2,.3,k),bounds=(.001,20),max_nfev=100);print('relax',z.x,np.mean(abs(z.fun)))
for name in ('a','b','c'):
 p=s['panel_'+name]; u=p['u']; w=np.roll(u,-1,axis=1)
 for key,v in p.items():
  if key=='u':continue
  a=D(u); b=ls(a,v)
  if name=='a':
   for i in range(8):
    r=np.linalg.norm(a@b-v,axis=1); mask=r<4*np.median(r); b=ls(a[mask],v[mask]);
  else: mask=np.ones(len(u),bool)
  print('PANEL',name,key,'n',sum(mask),'affine',np.round(b,5),'column rmse',np.mean((a[mask]@b-v[mask])**2,axis=0)**.5)
  if name=='a':
   for j in range(4):
    m=D(np.c_[u[:,j],w[:,j],u[:,j]*w[:,j],u[:,j]**3]); q=ls(m[mask],v[mask,j]);print(j,np.round(q,8),np.std((m@q-v[:,j])[mask]))
q=dict(s);q['train']={k:v[:52] if len(v)==72 else v for k,v in tr.items()};q['test']={'x':tr['x'][52:]}
for f in sorted(glob.glob('/tmp/hegel_continual_archive/*solver.py')):
 try:
  spec=importlib.util.spec_from_file_location('m',f);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);o=m.solve_batch(q);print(f.split('_')[-2], np.mean(abs(o['y_hat']-tr['y'][52:])),np.mean(abs(o['state_pred']-T[52:])))
 except Exception as e:print(type(e),e)
