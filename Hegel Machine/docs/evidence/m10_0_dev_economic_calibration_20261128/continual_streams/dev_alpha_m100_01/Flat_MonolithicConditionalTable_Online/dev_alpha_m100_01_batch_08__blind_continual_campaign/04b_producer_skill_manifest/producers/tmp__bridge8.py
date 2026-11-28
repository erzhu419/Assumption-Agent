import pickle,numpy as np
from itertools import combinations_with_replacement
from scipy.optimize import least_squares
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; cal=item['dataset']['calibration_load_cells']; l=cal['true_channel_loads']; raw=cal['bridge_raw']; s=cal['single_channel_sweeps']; x0=np.arange(4)[None,:]==s['lab_id'][:,None]; x0=x0*s['true_stress'][:,None]
for j in range(4):
 m=s['lab_id']==j; x=np.r_[l,x0[m]]; y=np.r_[raw[:,j],s['cmd_stress'][m]]; ok=abs(y)<6; x=x[ok];y=y[ok]; val=np.arange(len(y))%4==0
 print('\nCH',j,'isolated signals', (raw[:16,j].reshape(4,4)-np.median(raw[:16,j])).round(4)); u=x[:,j]
 for degree in [1,2,3,4,5,6]:
  for form in ['add','mixed'] if degree in [2,3] else ['add']:
   ids=[()] + [k for d in range(1,degree+1) for k in combinations_with_replacement(range(4),d) if form=='mixed' or k[0]==k[-1]]; a=np.stack([np.prod(x[:,k],axis=1) for k in ids],axis=1); v=np.linalg.lstsq(a[~val],y[~val],rcond=None)[0]; e=a@v-y
   print(degree,form,np.sqrt(np.mean(e[~val]**2)).round(6),np.sqrt(np.mean(e[val]**2)).round(6),[(k,round(z,5)) for k,z in zip(ids,v) if abs(z)>.002] if degree in [3,5] and form=='add' else '')
 for link in range(4):
  q=x[range(len(x)),link]
  for name in ['sin','cos','cubic']:
   def mat(f):
    v=np.sin(f[0]*q) if name=='sin' else np.cos(f[0]*q)-1 if name=='cos' else q**3
    return np.column_stack([np.ones(len(y)),x,u**3,v])
   def fun(f):
    a=mat(f);return (a@np.linalg.lstsq(a[~val],y[~val],rcond=None)[0]-y)[~val]
   fit=least_squares(fun,[2.],bounds=(.1,9.)); a=mat(fit.x); v=np.linalg.lstsq(a[~val],y[~val],rcond=None)[0]; e=a@v-y
   if np.sqrt(np.mean(e[val]**2)) < .005: print('nonlin',link,name,fit.x,v,np.sqrt(np.mean(e[val]**2)))
