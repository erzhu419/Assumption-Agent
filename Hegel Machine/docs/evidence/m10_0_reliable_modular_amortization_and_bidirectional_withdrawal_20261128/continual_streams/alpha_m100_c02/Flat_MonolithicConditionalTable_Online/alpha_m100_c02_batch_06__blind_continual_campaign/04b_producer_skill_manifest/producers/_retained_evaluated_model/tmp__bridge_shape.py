import pickle,numpy as np
from scipy.interpolate import PchipInterpolator
c=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']['calibration_load_cells']; l=c['true_channel_loads']; r=c['bridge_raw']; s=c['single_channel_sweeps']; np.set_printoptions(precision=4,suppress=True,linewidth=130)
for j in range(4):
 m=s['lab_id']==j; a=l[:,j]; own=(np.count_nonzero(l,axis=1)==1)&(a!=0); others=(np.count_nonzero(l,axis=1)==1)&(a==0)
 print('channel',j,'sweep',np.c_[s['true_stress'][m],s['cmd_stress'][m]].T,'isolated own',np.c_[a[own],r[own,j]].T)
 print('isolated leakage',[(k,np.c_[l[(l[:,k]!=0)&others,k],r[(l[:,k]!=0)&others,j]].T.tolist()) for k in range(4) if k!=j]); print('mixed raw quantiles',np.quantile(r[:,j],[0,.1,.5,.9,1])); print('largest raw rows',np.c_[l,r][np.argsort(abs(r[:,j]))[-3:]])
