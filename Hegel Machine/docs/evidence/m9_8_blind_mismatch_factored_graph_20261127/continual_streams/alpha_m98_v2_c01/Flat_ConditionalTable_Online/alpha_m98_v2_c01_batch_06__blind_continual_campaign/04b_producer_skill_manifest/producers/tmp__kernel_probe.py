import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 import chamber_probe as p
import numpy as np
from scipy.optimize import least_squares
m,d,tr,te,g,t,tau,theta,Y,R=p.m,p.d,p.tr,p.te,p.g,p.t,p.tau,p.theta,p.Y,p.R
labs=g['lab_id']; x=m._features(g,g)[0]; past,h,F=m._specimen(g,x,p.bridge); q=F@theta[2:]; E=Y-(h[:,None]*theta[:2])@R.T
r=E[:,-1]; print('loss floor from full horizon regressions:')
for name,columns in [('rate',[r]),('rate+constant',[r,np.ones_like(r)]),('rate+hold+constant',[r,h,np.ones_like(r)]),('rate+rate^2+hold+constant',[r,r*r,h,np.ones_like(r)])]:
 A=np.column_stack(columns); co=np.linalg.lstsq(A,E,rcond=None)[0];print(name,np.round(np.sqrt(np.mean((A@co-E)**2,axis=0)),4));print('kernels',np.round(co[:,[0,2,5,8,11,12]],4))
print('rate per lab:')
for l in np.unique(labs):
 j=labs==l;print(l,np.round(np.column_stack([q,r,h])[j],4),np.linalg.lstsq(np.column_stack([q[j],np.ones(sum(j))]),r[j],rcond=None)[0])
print('telemetry mean rows:');import json
for e in json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')):
 print(e['completed_batch_index'],np.round(np.mean(e['facility_telemetry_trace'],axis=0),2))
T=np.r_[t,1.];S=np.r_[-np.expm1(-t),1.]; z=np.r_[t+np.expm1(-t),0.]; Q=np.r_[t*t,0.]
for desc,A in [('q+intercept',np.column_stack([q,q*0+1])),('q+hold+intercept',np.column_stack([q,h,q*0+1])),('q+q2+hold+intercept',np.column_stack([q,q*q,h,q*0+1])),('features+hold',np.column_stack([F,h]))]:
 for bn,B in [('T,S',[T,S]),('T,S,z', [T,S,z]),('T,S,Q',[T,S,Q]),('T,S,R',[T,S,*R.T]),('T,S,Q,z',[T,S,Q,z])]:
  B=np.asarray(B); B=B[np.linalg.svd(B,full_matrices=False)[1].shape[0]*0:] # preserve basis ordering
  u,s,v=np.linalg.svd(B,full_matrices=False); B=v[s>1e-10]
  M=np.einsum('ni,bk->nkib',A,B).reshape(len(A),len(T),-1); pred=np.zeros_like(E)
  for l in np.unique(labs):
   j=labs==l; a=M[~j].reshape(-1,M.shape[-1]); y=E[~j].ravel(); co=np.linalg.lstsq(a,y,rcond=None)[0];pred[j]=M[j]@co
  e=pred-E;print(desc,bn,'pooled',np.round([np.sqrt(np.mean(e[:,:-1]**2)),np.sqrt(np.mean(e[:,-1]**2))],4),'worst',np.round(np.max([[np.sqrt(np.mean(e[labs==l,:-1]**2)),np.sqrt(np.mean(e[labs==l,-1]**2))] for l in np.unique(labs)],axis=0),4))
