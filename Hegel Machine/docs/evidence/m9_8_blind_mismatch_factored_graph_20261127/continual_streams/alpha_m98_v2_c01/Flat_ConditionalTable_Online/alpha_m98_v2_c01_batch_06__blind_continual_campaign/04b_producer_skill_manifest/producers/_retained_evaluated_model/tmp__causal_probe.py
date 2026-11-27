import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 import chamber_probe as p
import numpy as np
from scipy.optimize import least_squares
m,g,t,Y,R,theta=p.m,p.g,p.t,p.Y,p.R,p.theta
x=m._features(g,g)[0]; past,h,F=m._specimen(g,x,p.bridge); q=F@theta[2:]; E=Y-(h[:,None]*theta[:2])@R.T
T=np.r_[t,1.]; S=np.r_[-np.expm1(-t),1.]; labs=g['lab_id']
A=np.column_stack((q,q*q,h,np.ones(len(h)))); B=np.array([T,S]); M=np.einsum('ni,bk->nkib',A,B).reshape(len(A),len(T),-1)
co=np.linalg.lstsq(M.reshape(-1,8),E.ravel(),rcond=None)[0].reshape(4,2)
with open('/holdout_task/batch_solver.py','a') as f: f.write('\n\n_CHAMBER_COEFFICIENTS = np.array('+repr(co.tolist())+')\n')
print('material',p.tau,theta,'baseline creep',co)
benchC,benchR=m._branches(p.sp,p.dt,t,p.tau)
print('train strain/rate MAD',np.median(abs(benchC@theta[:2]+(p.sp[2]@theta[2:])[:,None]*t-p.observed(p.tr)[:,:-1])),np.median(abs(benchR@theta[:2]+p.sp[2]@theta[2:]-p.observed(p.tr)[:,-1])))
# Spatial inputs orthogonal to a free specimen-level creep amplitude identify the other mode.
def ker(kind,z):
 if kind=='exp':return np.r_[-np.expm1(-t*z)/z,1.]
 if kind=='log':return np.r_[np.log1p(t*z)/z,1.]
 if kind=='rat':return np.r_[t/(1+t*z),1.]
 if kind=='pow':return np.r_[t/np.sqrt(1+t*z),1.]
 if kind=='sqrt':return np.r_[2*t/(np.sqrt(1+t*z)+1),1.]
 if kind=='t':return T
C=np.c_[h,h*0+1]
for ka,kb in [('t','log'),('t','rat'),('sqrt','rat'),('t','pow'),('exp','rat'),('log','rat')]:
 def solve(z):
  a,b=ker(ka,z[0]),ker(kb,z[1]); ap=a-(a@b)/(b@b)*b; ep=E-(E@b)[:,None]/(b@b)*b
  coef=np.linalg.lstsq(C,ep@ap/(ap@ap),rcond=None)[0]; aamp=C@coef; bamp=(E-aamp[:,None]*a)@b/(b@b)
  return aamp[:,None]*a+bamp[:,None]*b-E,coef,bamp
 fit=least_squares(lambda z:solve(z)[0].ravel(),[.12,.8],bounds=(.0001,8)); e,coef,amp=solve(fit.x)
 print(ka,kb, np.round(fit.x,4),np.round(coef,4),'free amplitude RMSE',round(np.sqrt(np.mean(e**2)),5))
 if ka=='sqrt':
  print('lab q amplitude, hold and missing',np.round(np.c_[labs,q,amp,h,np.isnan(g['coupon_features_raw']).any(1)],4)); print('corr of free mode',np.round(np.corrcoef(np.c_[amp,q,q*q,x,h,past].T)[0],3))
