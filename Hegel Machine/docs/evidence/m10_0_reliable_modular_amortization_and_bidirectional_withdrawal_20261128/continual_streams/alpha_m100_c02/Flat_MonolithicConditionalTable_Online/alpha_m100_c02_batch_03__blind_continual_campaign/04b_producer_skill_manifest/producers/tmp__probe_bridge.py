import numpy as np,pickle
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
np.set_printoptions(precision=5,suppress=True,linewidth=130)
d=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; c=d['calibration_load_cells']; x=c['true_channel_loads']; y=c['bridge_raw']
for i in range(4):
 q=np.roll(x,-i,axis=1); a=np.column_stack([np.ones(len(q)), q]+[np.prod(q[:,idx],axis=1) for k in (2,3) for idx in combinations_with_replacement(range(4),k)])
 labels=['1','x','next','opposite','prev']+[str(idx) for k in (2,3) for idx in combinations_with_replacement(range(4),k)]
 for cols in (list(range(5)),list(range(15)),list(range(15))+[15],list(range(a.shape[1]))):
  A=a[:,cols]; b=least_squares(lambda b:A@b-y[:,i],np.linalg.lstsq(A,y[:,i],rcond=None)[0],loss='soft_l1',f_scale=.008).x; e=A@b-y[:,i]; good=abs(e)<.15
  if good.sum()>len(cols):
   b=np.linalg.lstsq(A[good],y[good,i],rcond=None)[0]; e=A@b-y[:,i]
  print(i,len(cols),np.quantile(abs(e),[.25,.5,.8,.9,.95,.975,1]),'bad',np.where(abs(e)>.15)[0], 'terms', [(labels[j],round(v,5)) for j,v in zip(cols,b) if abs(v)>.007])
 s=c['single_channel_sweeps']; sel=s['lab_id']==i; raw=s['cmd_stress'][sel]; true=s['true_stress'][sel]; print('single',np.c_[true,raw]);
print('split raw quantiles');
for name in ('train','test'):
 for key in ('past_channel_raw','hold_channel_raw'):
  print(name,key,np.quantile(d[name][key],[0,.01,.5,.99,1],axis=tuple(range(d[name][key].ndim-1))))
