import importlib.util, pickle, numpy as np
from pathlib import Path

def load(tag):
 p=Path('/tmp/hegel_continual_archive')/f'beta_m99_c02_{tag}_solver.py'
 spec=importlib.util.spec_from_file_location(tag,p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
a=load('batch_03'); c=s['sensor_calibration_sweeps']; x,y=c['ref_rosette'],c['raw_rosette']
np.set_printoptions(precision=6,suppress=True,linewidth=140)
for label, f in [('affine',lambda x,j:np.c_[np.ones(len(x)),x]),('archived polynomial',a._features)]:
 errors=np.empty_like(y); betas=[]
 for j in range(4):
  d=f(x,j)
  def fit(i):
   t=d[i]; v=y[i,j]; b=np.linalg.lstsq(t,v,rcond=None)[0]
   for _ in range(60):
    w=1/np.sqrt(1+((v-t@b)/.006)**2)
    b=np.linalg.lstsq(t*w[:,None],v*w,rcond=None)[0]
   clean=np.abs(v-t@b)<.015
   return np.linalg.lstsq(t[clean],v[clean],rcond=None)[0] if clean.sum()>=2*t.shape[1] else b
  for k in range(5):
   i=np.arange(len(x))%5==k; errors[i,j]=y[i,j]-d[i]@fit(~i)
  betas.append(fit(np.ones(len(x),bool)))
 print(label,'OOF abs error quantiles',np.quantile(abs(errors),[.5,.8,.9,.95,1],axis=0),'coefficients',np.array(betas),sep='\n')
 print('corrupt rows',np.flatnonzero(np.max(abs(errors),axis=1)>1))
ycal=a._calibrate(s)
print('calibrated observed range',ycal.min(),ycal.max())
for tag in ['s00','batch_01','batch_02','batch_03']:
 m=load(tag); S,U=m._responses(s); H,r=a._profile(S,ycal,s); R=s['source_ridge']; w=a._box_simplex(H.T@H+R*np.eye(H.shape[1]),H.T@r,s['u_box'],s['total_budget']); e=H@w-r
 print(tag,'weights',w,'shift RMSE',np.sqrt(np.mean(e.reshape(3,-1)**2,axis=1)),'bank RMSE',np.sqrt(np.mean(e.reshape(-1,4)**2,axis=0)))
