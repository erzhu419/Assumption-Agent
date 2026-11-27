import numpy as np,pickle
from itertools import combinations_with_replacement as comb
np.set_printoptions(precision=5,suppress=True,linewidth=140)
d=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; c=d['calibration_load_cells']; r=c['bridge_raw']; y=c['true_channel_loads']; n=len(y)
print('raw axial rows\n',np.c_[y,r][:16]); print('sweep size',n)
good=np.linalg.norm(r,axis=1)<10
print('excluded',np.flatnonzero(~good));
def basis(x,terms): return np.stack([x[...,t].prod(-1) for t in terms],axis=-1)
for deg in (1,2,3,4):
 for additive in (True,False):
  ts=[t for k in range(deg+1) for t in comb(range(4),k) if not additive or len(set(t))<=1]; a=basis(y,ts); p=np.linalg.lstsq(a[good],r[good],rcond=None)[0]; e=a@p-r; cv=[]
  for f in range(5):
   test=good&(np.arange(n)%5==f); train=good&~test
   cv.extend(a[test]@np.linalg.lstsq(a[train],r[train],rcond=None)[0]-r[test]);
  print(deg,additive,len(ts),'rmse',np.mean(e[good]**2)**.5,'cv',np.mean(np.array(cv)**2)**.5,'common shift',e[~good].round(4))
  if deg==3 and not additive:
   print(list(zip(ts,p.round(5).tolist())))
