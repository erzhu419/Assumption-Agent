import numpy as np
from scipy.optimize import least_squares
from ros_probe import x,y,fit
from ros_joint_input import bank
valid=fit(x,y,'affine')[1]; bi=bank['bank_id']; sz=bank['ref_signal']; sy=bank['raw_readout']
# One normalized shape, one bias and gain per bank; test the residual own-channel correction.
def funs(a,b,k,mode):
 d=k[0]*a+np.tanh(k[1]*a)+k[2]*b+k[3]*(1-np.cos(k[4]*b))
 if mode=='rational': d+=k[5]*a/(1+k[6]*a*a)
 if mode=='sine': d+=k[5]*np.sin(k[6]*a)
 if mode=='cubic': d+=k[5]*a**3
 if mode=='tanh': d+=k[5]*np.tanh(k[6]*a)
 if mode=='asinh': d+=k[5]*np.arcsinh(k[6]*a)
 return d
lo=[-.1,.6,-.4,-.3,.5,-2,.1]; hi=[.9,3.,.5,.3,4.,2.,5.]
for m in ('none','rational','sine','cubic','tanh','asinh'):
 es=[]
 for ids in [np.array([],int),*np.array_split(np.argsort(np.linalg.norm(x,axis=1)),5)]:
  tr=valid.copy(); tr[ids]=False; va=ids[valid[ids]] if ids.size else np.flatnonzero(valid)
  n=5 if m=='none' else (6 if m=='cubic' else 7)
  def res(p):
   k=p[:n]; offs=p[n:n+4]; g=p[n+4:];
   return np.r_[(offs + g*funs(x[tr],x[tr][:,[2,3,0,1]],k,m)-y[tr]).ravel(),(offs[bi]+g[bi]*funs(sz,0.,k,m)-sy)]
  z=least_squares(res,[.19,1.4,.19,-.06,2.,.03,1.][:n]+[.07,-.06,.08,-.05,1.4,.9,1.4,.9],bounds=(lo[:n]+[-2]*4+[.05]*4,hi[:n]+[2]*4+[4]*4),loss='cauchy',f_scale=.006,max_nfev=300).x
  r=z[n:n+4]+z[n+4:]*funs(x[va],x[va][:,[2,3,0,1]],z[:n],m)-y[va]; es.append(np.sqrt(np.mean(r*r)))
  if not ids.size: p=z
 print(m,np.round(es,7),p.round(5))
