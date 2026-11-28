import numpy as np, pickle
from itertools import combinations_with_replacement
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; c=p['calibration_load_cells']; x=c['true_channel_loads']; y=c['bridge_raw']; r=y-x
np.set_printoptions(precision=5,suppress=True,linewidth=130)
print('rows',np.c_[x,y]); print('spikes',np.flatnonzero(np.max(abs(r),axis=1)>2)); print('split ranges',[(s,k,np.min(p[s][k],axis=tuple(range(p[s][k].ndim-1))),np.max(p[s][k],axis=tuple(range(p[s][k].ndim-1)))) for s in ('train','test') for k in ('past_channel_raw','hold_channel_raw')]);
keep=np.max(abs(r),axis=1)<2
for label, terms in [('linear',[(i,) for i in range(4)]),('diagonal-3',[(i,)*q for q in range(1,4) for i in range(4)]),('quad',[z for q in (1,2) for z in combinations_with_replacement(range(4),q)]),('quad+diag3',[z for q in (1,2) for z in combinations_with_replacement(range(4),q)]+[(i,)*3 for i in range(4)]),('cubic',[z for q in (1,2,3) for z in combinations_with_replacement(range(4),q)])]:
 a=np.column_stack([np.ones(len(x))]+[np.prod(x[:,z],axis=1) for z in terms]); a=a[keep]; b=r[keep]; pi=np.linalg.pinv(a); beta=pi@b; err=b-a@beta; cv=err/(1-np.sum(a*pi.T,axis=1))[:,None]; print(label,'rmse',np.sqrt(np.mean(cv**2,axis=0)),'worst',np.max(abs(cv),axis=0),'coeff',beta if label in ('quad+diag3',) else '')
