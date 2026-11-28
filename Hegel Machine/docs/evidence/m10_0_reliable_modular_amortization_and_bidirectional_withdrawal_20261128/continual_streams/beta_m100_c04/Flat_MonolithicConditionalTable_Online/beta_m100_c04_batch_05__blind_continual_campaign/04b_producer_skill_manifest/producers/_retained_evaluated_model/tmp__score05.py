import sys,pickle,numpy as np
sys.path.insert(0,'/holdout_task')
import batch_solver as m
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; x,y=c['ref_rosette'],c['raw_rosette']; clean=(abs(y)<8).all(1)
for d in [6,8,10,12,14]:
 r=[]
 for j in np.flatnonzero(clean):
  h=np.arange(len(x))==j
  p,*_=m._cal_fit(c,h,cross_degree=d); r.append(p(x[j])[0]-y[j]);
 print(d,'oof RMS/max',np.sqrt(np.mean(np.square(r),axis=0)),np.max(abs(np.array(r)),axis=0))
p,b,q,co=m._cal_fit(c,cross_degree=8); rem=(y-p(x))[clean]; X=x[clean]; print('self',q,'cross',np.round(co,6)); print('insample',np.std(rem,axis=0));
for a in range(4):
 v=X[:,(np.arange(4)+a)%4]; allq=[q for d in [1,2,3,4] for q in combinations_with_replacement(range(4),d) if q not in [(0,)*d,(2,)*d]]; G=np.column_stack([np.prod(v[:,q],axis=1) for q in allq]); print('bank',a,sorted([(abs(z),q,z) for z,q in zip(np.corrcoef(np.c_[rem[:,a],G].T)[0,1:],allq)],reverse=True)[:6]); print('cross profile',np.c_[v[:,2],(y-p(x))[clean,a]][np.argsort(v[:,2])][::5].round(5).tolist())
print('obs cal limits',np.min(m._calibrate(s)),np.max(m._calibrate(s))); print('shapes',s['source_modes'].shape,s['sensor_ij'].shape); print('sensor ij',s['sensor_ij'].tolist())
