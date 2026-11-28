import pickle,numpy as np
from itertools import combinations_with_replacement as C
np.set_printoptions(precision=4,suppress=True,linewidth=140)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def pol(x,d):return np.column_stack([np.ones(len(x))]+[np.prod(x[:,j],1) for k in range(1,d+1) for j in C(range(x.shape[1]),k)])
def robust(f,y,it=15):
 b=np.linalg.lstsq(f,y,rcond=None)[0]
 for _ in range(it):
  er=np.linalg.norm(f@b-y,axis=1); k=np.argsort(er)[:max(f.shape[1],int(.8*len(f)))];b=np.linalg.lstsq(f[k],y[k],rcond=None)[0]
 return b
for name in ('panel_a','panel_b','panel_c'):
 p=s[name]; u=p['u'];print(name)
 for key,v in p.items():
  if key=='u':continue
  print(key,'range',v.min(0),v.max(0))
  for d in [1,2,3]:
   f=pol(u,d); b=robust(f,v) if name=='panel_a' else np.linalg.lstsq(f,v,rcond=None)[0];e=np.linalg.norm(f@b-v,axis=1);print(d,np.quantile(e,[0,.25,.5,.75,.9,1]));print(b if d==1 or (name=='panel_a' and d==2) else '')
  if name=='panel_a':print('linear residual per row',np.column_stack((u,v-pol(u,1)@robust(pol(u,1),v))))
z=s['train']['traj']; t=s['eval_times'];print('svd',np.linalg.svd(z,compute_uv=False));print('state range',z.min(0),z.max(0)); print('examples x,y,z',np.column_stack((s['train']['x'], s['train']['y'],z))[:22]);print('state increment ratios', (np.diff(z)[:,1:]/np.diff(z)[:,:-1])[:22]);print('corr final/y',np.corrcoef(s['train']['y'],z[:,-1]))
