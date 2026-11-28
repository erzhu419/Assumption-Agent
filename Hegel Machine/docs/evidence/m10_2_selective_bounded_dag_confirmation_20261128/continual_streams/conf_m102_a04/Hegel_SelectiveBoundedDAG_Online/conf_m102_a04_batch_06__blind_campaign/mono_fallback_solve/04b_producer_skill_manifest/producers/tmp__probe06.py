import pickle, numpy as np
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
import importlib.util
p='/tmp/hegel_continual_archive/conf_m102_a04_batch_00_solver.py'; spec=importlib.util.spec_from_file_location('prior',p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=5,suppress=True,linewidth=135)
ls=lambda a,b:np.linalg.lstsq(a,b,rcond=1e-10)[0]
poly=lambda x,d:np.column_stack([np.ones(len(x))]+[x[:,ix].prod(1) for j in range(1,d+1) for ix in combinations_with_replacement(range(x.shape[1]),j)])
for k in ('panel_a','panel_b','panel_c'):
 p=s[k]; u=p['u']; print(k,'u bounds',u.min(0),u.max(0));
 for name,v in p.items():
  if name=='u': continue
  print(name,'samples',np.c_[u[:5],v[:5]],'std',v.std(0));
  for d in (1,2):
   f=poly(u,d); r=v-f@ls(f,v); print('degree',d,'mean abs',abs(r).mean(0),'median abs',np.median(abs(r),axis=0))
  print('linear coefs',ls(poly(u,1),v)); print('diagonal alignment',m._align({'u':u,'v':v}))
o,g=m._align(s['panel_a']); x=s['train']['x']; z=(x-o)/g; y=s['train']['y']; tr=s['train']['traj']; t=s['eval_times']; print('telemetry',s['telemetry_vec'],'t',t,'summary',s['train']['summary']); print('x/z stats',x.min(0),x.max(0),z.min(0),z.max(0)); print('y/traj samples',np.c_[x[:10],y[:10],tr[:10]]); print('traj SVD',np.linalg.svd(tr,compute_uv=False)); print('centered SVD',np.linalg.svd(tr-tr.mean(0),compute_uv=False)); B=m._time_basis(t,tr[:52]); print('archive time basis max/mae',abs(B@ls(B,tr.T)-tr.T).max(),abs(B@ls(B,tr.T)-tr.T).mean()); print('y vs traj',ls(poly(tr,1),y),abs(poly(tr,1)@ls(poly(tr,1),y)-y).mean());
for label,w in [('raw',x),('aligned',z),('reverse',x*g+o),('panel_c',m._panel_maps(s['panel_c'],z)),('panel_c_raw',m._panel_maps(s['panel_c'],x))]:
 T=np.c_[y,tr]; print(label)
 for d in (1,2,3):
  f=poly(w,d); sd=np.maximum(f[:52].std(0),1e-7); f=f/sd; f[:,0]=1; b=ls(np.r_[f[:52],np.eye(f.shape[1])*.025],np.r_[T[:52],np.zeros((f.shape[1],9))]); er=abs(f@b-T); print(d,'train',er[:52].mean(0)[[0,1,8]],'val',er[52:].mean(0)[[0,1,8]])
