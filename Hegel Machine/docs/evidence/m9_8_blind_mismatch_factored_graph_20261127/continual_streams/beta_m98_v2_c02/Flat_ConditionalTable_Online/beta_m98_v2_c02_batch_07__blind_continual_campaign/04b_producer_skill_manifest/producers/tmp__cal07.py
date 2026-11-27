import pickle,numpy as np
from itertools import combinations_with_replacement
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=item['study']; q=s['sensor_calibration_sweeps']; x,y=q['ref_rosette'],q['raw_rosette']
print('ranges',x.min(0),x.max(0),y.min(0),y.max(0)); print('single sweeps', {k:np.asarray(v).reshape(4,6).round(5) for k,v in q['single_bank_sweeps'].items()})
sets=[('affine',[()]+[(i,) for i in range(4)]),('archive',[()]+[(i,) for i in range(4)]+[(i,)*3 for i in range(4)]+[tuple(sorted((i,(i+1)%4))) for i in range(4)])]; sets+= [('degree '+str(n),[()] + [a for d in range(1,n+1) for a in combinations_with_replacement(range(4),d)]) for n in (2,3,4)]
for name,terms in sets:
 D=np.array([x[:,a].prod(1) for a in terms]).T; err=[]
 for test in np.array_split(np.random.default_rng(42).permutation(len(x)),5):
  tr=np.setdiff1d(np.arange(len(x)),test); c=np.linalg.lstsq(D[tr],y[tr],rcond=None)[0]; et=abs(D[tr]@c-y[tr]).max(1); keep=et<=max(5*np.median(et),1e-10); c=np.linalg.lstsq(D[tr][keep],y[tr][keep],rcond=None)[0]; err.extend(abs(D[test]@c-y[test]).max(1));
 print(name,'OOF row max percentiles',np.quantile(err,[0,.5,.9,.95,1]));
 if name=='archive':
  c=np.linalg.lstsq(D,y,rcond=None)[0]; e=abs(D@c-y).max(1); keep=e<=max(5*np.median(e),1e-10); c=np.linalg.lstsq(D[keep],y[keep],rcond=None)[0]; print('archive coef',c.round(7),'removed',np.flatnonzero(~keep),'inlier max',abs(D[keep]@c-y[keep]).max())
print('sensor locations',s['sensor_ij'].T)
