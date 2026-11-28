import numpy as np,pickle
from scipy.optimize import least_squares
from scipy.special import erf
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; X,Y=c['ref_rosette'],c['raw_rosette']; q=c['single_bank_sweeps']; i=q['bank_id']; t=q['ref_signal']; v=q['raw_readout']; clean=np.max(abs(Y),axis=1)<4; axis=np.count_nonzero(X,axis=1)==1; z=np.max(abs(Y),axis=1)<4
for j in range(4):
 off=np.median(Y[clean&axis&(X[:,j]==0)&(X[:,(j+2)%4]==0),j]); m=clean&axis&(X[:,j]!=0); k=(i==j)&(abs(v)<4); a=np.r_[X[m,j],t[k]]; b=np.r_[Y[m,j],v[k]]-off
 print('\nbank',j,'offset',off)
 for name,f in [('tanh',np.tanh),('erf',erf),('atan',np.arctan),('sqrt',lambda x:x/(1+x*x)**.5),('rat',lambda x:x/(1+x*x))]:
  for extra in [0,1,2]:
   def pred(p):
    d=[f(np.exp(p[0])*(a+p[1]))-f(np.exp(p[0])*p[1]),a]; d.extend([a*a,a**3][:extra]); return np.array(d).T
   def resid(p):
    D=pred(p); return D@np.linalg.lstsq(D,b,rcond=None)[0]-b
   p=least_squares(resid,[0,0],bounds=([-2,-1],[3,1]),gtol=1e-12).x; D=pred(p); co=np.linalg.lstsq(D,b,rcond=None)[0]; err=resid(p); print(name,extra,'max',round(max(abs(err)),7), 'p',p.round(5),'coef',co.round(5))
 print('opposite axis',np.c_[X[clean&axis&(X[:,(j+2)%4]!=0),(j+2)%4],Y[clean&axis&(X[:,(j+2)%4]!=0),j]-off].round(6))
 print('mixed sample',np.c_[X[~axis][:3],Y[~axis][:3,j]])
