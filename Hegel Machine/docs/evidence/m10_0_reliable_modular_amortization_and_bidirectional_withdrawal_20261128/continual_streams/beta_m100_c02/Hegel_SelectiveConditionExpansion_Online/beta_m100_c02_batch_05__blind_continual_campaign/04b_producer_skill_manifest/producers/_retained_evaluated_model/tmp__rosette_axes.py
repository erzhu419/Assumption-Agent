import pickle,numpy as np
from scipy.optimize import least_squares
c=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']['sensor_calibration_sweeps']; d=c['single_bank_sweeps']; np.set_printoptions(precision=5,suppress=True)
def f(z,k,kind):
 if kind=='tanh': return np.tanh(k*z)
 if kind=='root': return k*z/np.sqrt(1+(k*z)**2)
 if kind=='atan': return np.arctan(k*z)
 if kind=='asinh': return np.arcsinh(k*z)
 if kind=='erf':
  from scipy.special import erf
  return erf(k*z)
for kind in ('tanh','root','atan','asinh','erf'):
 for extra in ('none','linear','shift','both'):
  rows=[]
  for j in range(4):
   q=d['bank_id']==j; z=d['ref_signal'][q]; y=d['raw_readout'][q]; ok=np.abs(y)<4
   def pred(p): return p[0]+p[1]*f(z+(p[3] if extra in ('shift','both') else 0),p[2],kind)+(p[-1]*z if extra in ('linear','both') else 0)
   p=least_squares(lambda p:(pred(p)-y)[ok], [0,1,1]+[0]*(extra!='none')+[0]*(extra=='both'),max_nfev=500).x
   rows.append([np.sqrt(np.mean((pred(p)-y)[ok]**2)),*p])
  print(kind,extra,np.array(rows).round(6).tolist())
