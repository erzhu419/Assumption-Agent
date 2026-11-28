import pickle, numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; d=s['sensor_calibration_sweeps']; X=d['ref_rosette']; Y=d['raw_rosette']; ok=(abs(Y)<5).all(1); bank=d['single_bank_sweeps']; print('large rows',np.where(~ok)[0]);
def basis(x,k,kind,e):
 z=k*x
 if kind=='tanh': f=np.tanh(z)
 if kind=='rat': f=z/(1+z*z)
 if kind=='root': f=z/np.sqrt(1+z*z)
 if kind=='atan': f=np.arctan(z)
 if kind=='erf':
  from scipy.special import erf
  f=erf(z)
 if kind=='poly': f=x**3
 terms=[np.ones(len(x)),x,x*x,f];
 if e=='f2': terms+=[f*f]
 if e=='rat2': terms+=[x*x/(1+z*z)]
 return np.column_stack(terms)
for j in range(4):
 sel=(bank['bank_id']==j)&(abs(bank['raw_readout'])<5); x=bank['ref_signal'][sel]; y=bank['raw_readout'][sel]; print('bank',j,np.round(y,6));
 for kind in ['tanh','rat','root','atan','erf']:
  for e in ['', 'f2','rat2']:
   def fun(q):
    a=basis(x,np.exp(q[0]),kind,e); return a@np.linalg.lstsq(a,y,rcond=None)[0]-y
   res=least_squares(fun,[0.],bounds=(-2,2),gtol=1e-13); print(kind,e,round(np.exp(res.x[0]),5),round(np.linalg.norm(res.fun),7),end='; ')
  print()
for deg in range(3,10):
 A=np.column_stack([np.ones(len(X))]+[X**i for i in range(1,deg+1)]); a=A[ok]; y=Y[ok]; b=np.linalg.lstsq(a,y,rcond=None)[0]; r=y-a@b; q=np.linalg.qr(a,mode='reduced')[0]; cv=r/(1-(q*q).sum(1))[:,None]; print('poly',deg,'rmse',np.sqrt(np.mean(r*r)),'loo',np.sqrt(np.mean(cv*cv)), 'max norm',np.max(np.linalg.norm(cv,axis=1))); 
 if deg in (5,8): print(np.round(b,5))
