import pickle,numpy as np
from itertools import combinations_with_replacement
c=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']['calibration_load_cells']
a,y=map(np.asarray,(c['bridge_raw'],c['true_channel_loads']))
solo=np.count_nonzero(y,axis=1)==1
part=np.empty(len(y),int)
for idx in [np.flatnonzero(solo),np.flatnonzero(~solo)]: part[idx]=np.arange(len(idx))%4
for raw,true,tag in [(a,y,'inverse'),(y,a,'forward')]:
 for kind in ('affine','square','quadratic','diagonal_cubic'):
  z=np.c_[np.ones(len(raw)),raw]
  if kind=='square': z=np.c_[z,raw**2]
  if kind=='quadratic': z=np.c_[z,*[raw[:,i]*raw[:,j] for i,j in combinations_with_replacement(range(4),2)]]
  if kind=='diagonal_cubic': z=np.c_[z,raw**2,raw**3]
  e=np.empty_like(true)
  for fold in range(4):
   m=part==fold; e[m]=z[m]@np.linalg.lstsq(z[~m],true[~m],rcond=None)[0]-true[m]
  print(tag,kind,'channel rms by sweep',*[round(np.mean(e[m]**2)**.5,8) for m in (solo,~solo)],'sum rms',round(np.mean(e.sum(1)**2)**.5,8))
