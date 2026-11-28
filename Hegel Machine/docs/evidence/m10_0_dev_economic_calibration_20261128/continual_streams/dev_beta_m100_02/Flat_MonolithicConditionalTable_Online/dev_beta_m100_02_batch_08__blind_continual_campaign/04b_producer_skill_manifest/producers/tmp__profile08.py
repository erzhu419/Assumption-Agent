import pickle,numpy as np
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['sensor_calibration_sweeps']; b=t['single_bank_sweeps']
families={'tanh':np.tanh,'atan':np.arctan,'sin':np.sin,'asinh':np.arcsinh,'rat':lambda z:z/(1+z*z),'softsign':lambda z:z/np.sqrt(1+z*z),'erf':__import__('scipy.special',fromlist=['erf']).erf}
for name,fn in families.items():
 def profile(f,ret=False):
  cs=[]; rs=[]
  for j in range(4):
   ix=(b['bank_id']==j)&(abs(b['raw_readout'])<4); x,y=b['ref_signal'][ix],b['raw_readout'][ix]; D=np.column_stack([np.ones(len(x)),x,fn(f*x)]); c=np.linalg.lstsq(D,y,rcond=None)[0];cs.append(c);rs.extend(D@c-y)
  return (np.array(cs),np.max(abs(np.array(rs)))) if ret else np.mean(np.square(rs))
 opt=minimize_scalar(profile,bounds=(.3,4.),method='bounded',options={'xatol':1e-13}); print(name,'freq',opt.x,'RMSE',opt.fun**.5,'max',profile(opt.x,True)[1],'coeff',np.round(profile(opt.x,True)[0],5))
 if name=='tanh':
  c=profile(opt.x,True)[0]; x=t['ref_rosette']; r=t['raw_rosette']-(c[:,0]+c[:,1]*x+c[:,2]*fn(opt.x*x)); good=np.all(abs(t['raw_rosette'])<4,axis=1); x=x[good];r=r[good]; print('joint remaining RMSE',np.mean(r*r,axis=0)**.5); print('cross regression',np.round(np.linalg.lstsq(x,r,rcond=None)[0],4))
