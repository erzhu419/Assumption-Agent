import pickle
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
from calibrated import calibrate,inverse,time_modes,design
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; q=tr['traj']; z=inverse(tr['x'],calibrate(s['panel_a'])); n=len(z)
D=design(z); Q=PolynomialFeatures(2).fit_transform(z); Ys=tr['y']; splits=[(np.arange(52),np.arange(52,72))]+list(KFold(4,shuffle=True,random_state=111).split(z))
models=[]
for i in range(4):
 for variant in ['whole','affine','quad','scalar','linear','pair','add']:
  def basis(k,i=i,variant=variant):
   x=z[:,i]; inv=1/(1+k*x*x)
   if variant=='whole': return Q*inv[:,None]
   if variant=='affine': return np.c_[D,D[:,1:]*inv[:,None]]
   if variant=='quad': return np.c_[Q,D*inv[:,None]]
   if variant=='scalar': return np.c_[Q,inv,x*inv]
   if variant=='linear': return np.c_[D,Q[:,5:]*inv[:,None]]
   if variant=='pair': return np.c_[Q,(z[:,[j for j in range(4) if j!=i]]*x[:,None])*inv[:,None]]
   return np.c_[Q,np.sin(k*x),np.cos(k*x)]
  models.append((f'{i}:{variant}',basis,(.08,4.) if variant!='add' else (.7,5.)))
for i in range(4):
 for j in range(i):
  for sign in [-1,1]:
   def basis(k,i=i,j=j,sign=sign):
    inv=1/(1+k*(z[:,i]+sign*z[:,j])**2);return np.c_[Q,D*inv[:,None]]
   models.append((f'{i},{j},{sign}:quad',basis,(.08,4.)))
for yid in range(3):
 result=[]
 for name,basis,bounds in models:
  errors=[]; fits=[]; pars=[]
  for a,b in splits:
   h,m=time_modes(s['eval_times'],q[a],len(a)); y=Ys if yid==0 else (q@np.linalg.pinv(h.T))[:,yid-1]
   def fit(k):
    F=basis(k); W=np.linalg.lstsq(F[a],y[a],rcond=1e-9)[0]; r=F[a]@W-y[a]; return np.mean(r*r), F@W
   o=minimize_scalar(lambda k:fit(k)[0],bounds=bounds,method='bounded'); mse,p=fit(o.x); errors.append(np.mean(abs(p[b]-y[b]))); fits.append(mse**.5);pars.append(o.x)
  result.append((name,np.round(errors,4),np.round(fits,4),np.round(pars,3)))
 print('target',yid)
 for row in sorted(result,key=lambda r:np.mean(r[1]))[:14]:print(*row)
