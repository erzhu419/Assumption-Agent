import pickle
import numpy as np
from itertools import combinations_with_replacement
np.set_printoptions(precision=6,suppress=True,linewidth=150)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=i['dataset']; c=d['calibration_load_cells']; x=c['true_channel_loads']; y=c['bridge_raw']; idx=np.arange(len(x)); clean=np.max(np.abs(y),1)<5
print('large raw rows',idx[~clean]); print('large row x and raw',np.c_[x[~clean],y[~clean]])
base=np.c_[np.ones(len(x)),x]; q=np.column_stack([x[:,k]*x[:,l] for k,l in combinations_with_replacement(range(4),2)]); cubic=np.column_stack([x[:,k]*x[:,l]*x[:,m] for k,l,m in combinations_with_replacement(range(4),3)])
ytotal=y.sum(1,keepdims=True)
for name,a in [('linear',base),('linear+diagonal cubic',np.c_[base,x**3]),('linear+diagonal quad/cubic',np.c_[base,x**2,x**3]),('quadratic+diagonal cubic',np.c_[base,q,x**3]),('full cubic',np.c_[base,q,cubic])]:
 for yy,eligible in [(y,clean),(ytotal,np.ones(len(x),bool))]:
  err=np.zeros_like(yy); b=np.linalg.lstsq(a[eligible],yy[eligible],rcond=None)[0]
  for j in range(5):
   tr=eligible&(idx%5!=j); te=eligible&(idx%5==j); err[te]=a[te]@np.linalg.lstsq(a[tr],yy[tr],rcond=None)[0]-yy[te]
  print(name,'per-channel clean' if yy is y else 'raw total all', 'OOF RMSE',np.sqrt(np.mean(err[eligible]**2,0)), 'max',np.max(abs(err[eligible]),0))
  if name=='linear+diagonal cubic' and yy is y:
   print('coefficients:',b); print('large row residuals:',yy[~clean]-a[~clean]@b)
print('single-channel command sweeps'); s=c['single_channel_sweeps']; print(np.c_[s['lab_id'],s['true_stress'],s['cmd_stress']])
