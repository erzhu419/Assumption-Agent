import importlib.util, pickle
import numpy as np
spec=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; sweeps=s['sensor_calibration_sweeps']; x,y=map(np.asarray,(sweeps['ref_rosette'],sweeps['raw_rosette'])); bank=sweeps['single_bank_sweeps']; row=np.arange(len(x));
for name in ('sequential','joint','pair_difference','rational'):
 errors=[]
 for fold in range(5):
  train=row%5!=fold; sw=dict(sweeps,ref_rosette=x[train],raw_rosette=y[train]); p,b=m.calibration_map(sw); F=[]
  for j in range(4):
   t,u,v=x[:,j],x[:,(j-1)%4],x[:,(j+1)%4]; A=np.column_stack((np.ones(len(x)),t,t**3,u,v,t*v)); base=A@np.r_[p[:,j],b[:,j]]; keep=abs(y[:,j]-base)<1e-3
   if name!='sequential':
    sel=np.asarray(bank['bank_id'])==j; z=np.asarray(bank['ref_signal'])[sel]; reading=np.asarray(bank['raw_readout'])[sel]; B=np.column_stack((np.ones(len(z)),z,z**3,np.zeros((len(z),3)))); ok=abs(B@np.r_[p[:,j],b[:,j]]-reading)<1e-3
    if name=='pair_difference':
     A=np.column_stack((A,(u-v)**2,t*(u-v)**2)); B=np.column_stack((B,np.zeros((len(z),2))))
    if name=='rational':
     # Leading correction from dividing the fitted transfer by 1+a*Q^2.
     A=np.column_stack((A,-base*t**2,-base*(u-v)**2)); B=np.column_stack((B,-(B@np.r_[p[:,j],b[:,j]])*z**2,np.zeros(len(z))))
    use=train&keep; coef=np.linalg.lstsq(np.vstack((A[use],B[ok])),np.r_[y[use,j],reading[ok]],rcond=None)[0]; base=A@coef
   e=(base-y[:,j])[~train&keep]; errors.append((np.sqrt(np.mean(e**2)),len(e)))
 print(name,'worst bank-fold RMSE',max(v for v,n in errors),'pooled held-out RMSE',np.sqrt(sum(v*v*n for v,n in errors)/sum(n for v,n in errors)),'retained',sum(n for v,n in errors))
