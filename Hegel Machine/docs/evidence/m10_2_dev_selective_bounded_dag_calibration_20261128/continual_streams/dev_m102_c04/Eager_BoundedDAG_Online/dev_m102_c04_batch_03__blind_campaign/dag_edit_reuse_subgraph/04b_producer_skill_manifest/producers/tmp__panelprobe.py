import numpy as np,pickle,runpy
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=4,suppress=True)
for name in ('panel_a','panel_b','panel_c'):
 p=s[name]; u=p['u']; a=np.c_[np.ones(len(u)),u]; print(name,'u',u.min(0),u.max(0))
 for k,v in p.items():
  if k=='u':continue
  w=np.linalg.lstsq(a,v,rcond=None)[0]; e=np.linalg.norm(a@w-v,axis=1); print(k,w,'error',np.sort(e))
for k in ('telemetry_vec','eval_times'):print(k,s[k]);
for k,v in s['train'].items():print(k,np.shape(v),v[:3]);
