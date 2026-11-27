import glob, importlib.util, pickle
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import diags
sp=importlib.util.spec_from_file_location('old',glob.glob('/tmp/hegel_continual_archive/*solver.py')[0]); old=importlib.util.module_from_spec(sp); sp.loader.exec_module(old)
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=p['study']; c=s['sensor_calibration_sweeps']; ref=c['ref_rosette']; raw=c['raw_rosette']; d=p['campaign_diagnostics']
X=np.c_[np.ones(len(ref)),ref]; fit=np.linalg.lstsq(X,raw,rcond=None)[0]
print('cal samples/ranges',ref.shape,ref.min(0),ref.max(0),'affine coeff',fit,'affine rmse',np.sqrt(np.mean((raw-X@fit)**2,axis=0)),sep='\n')
print('sensor edge counts',[(s['sensor_ij'][:,j]==i).sum() for j in range(2) for i in (0,(s['nx'],s['ny'])[j]-1)])
Y=np.stack([np.linalg.solve(fit[1:].T,(q['y_pulse_raw'].reshape(-1,4)-fit[0]).T).T.ravel() for q in d],axis=1); W=np.stack([q['w_pulse'] for q in d],axis=1)
axis=old._axis
# Each edge has a normalized inward conductance and diagonal row sum; optional opposite-wall coupling.
def forward(v):
 it=iter(v.reshape(4,-1))
 def ax(kap,vel,length,rob):
  A=axis(kap,vel,length,rob).tolil(); h=length/(len(kap)-1)
  for i,j,k in [(0,1,-1),(-1,-2,0)]:
   b=next(it); f,r=b[:2]; e=b[2] if len(b)>2 else 0
   A[i,j]=-f*kap[i]/h**2; A[i,k]=-e*kap[i]/h**2; A[i,i]=(f+e)*kap[i]/h**2+r*rob/h
  return A.tocsc()
 old._axis=ax
 try: return old._forward(s)
 finally: old._axis=axis
for width in [2,3]:
 start=np.tile([1,1]+[0]*(width-2),4)
 def err(v): return (forward(v)[0]@W-Y)
 print('baseline rmse by shift and pulse',np.sqrt(np.mean(err(start).reshape(3,-1,2)**2,axis=1)))
 for cols in [slice(None),[0],[1]]:
  out=least_squares(lambda v:err(v)[:,cols].ravel(),start,diff_step=1e-4,gtol=1e-13,max_nfev=150)
  print('cols,width',cols,width,'fit',out.x.reshape(4,width),'rmse',np.sqrt(np.mean(err(out.x).reshape(3,-1,2)**2,axis=1)),sep='\n')
