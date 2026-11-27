import contextlib, io
with contextlib.redirect_stdout(io.StringIO()):
 import chamber_probe as p
import numpy as np
from scipy.optimize import least_squares
m,d,tr,te,g,t,dt,tau,theta,Y,R=p.m,p.d,p.tr,p.te,p.g,p.t,p.dt,p.tau,p.theta,p.Y,p.R
T=np.r_[t,1.]; Q=np.r_[t*t,0.]; Z=np.r_[np.log1p(t),1.]; S=np.r_[-np.expm1(-t),1.]
xs={'diag':m._features(g,g)[0], 'train+diag':m._features(tr,g)[1], 'test+diag':m._features(te,g)[1], 'all':m._features(p.joined([tr,te]),g)[1]}
for name,x in xs.items():
 past,hold,F=m._specimen(g,x,p.bridge); drift=F@theta[2:]; base=(hold[:,None]*theta[:2])@R.T; E=Y-base
 print('\n',name,'drift range',np.round(np.sort(drift),3),'singular values after reset',np.round(np.linalg.svd(E,compute_uv=False)[:5],4))
 print('drift, inferred rate, residuals at t_0,t_mid,t_end,lab,missing')
 if name in ('diag','all'):
  ix=np.argsort(drift);print(np.round(np.column_stack((drift,E[:,-1],E[:,[0,5,-2]],g['lab_id'],np.isnan(g['coupon_features_raw'][:,0])))[ix],4))
 def score(basis):
  A=np.stack([a[:,None]*b for a,b in basis],axis=-1)
  P=np.empty_like(E)
  for lab in np.unique(g['lab_id']):
   v=g['lab_id']==lab; ac=A[~v]; ec=E[~v]; co=least_squares(lambda c:(ac@c-ec).ravel(),np.linalg.lstsq(ac.reshape(-1,A.shape[-1]),ec.ravel(),rcond=None)[0],loss='soft_l1',f_scale=.02).x;P[v]=A[v]@co
  E2=P-E
  return np.round([np.sqrt(np.mean(E2[:,:-1]**2)),np.sqrt(np.mean(E2[:,-1]**2)),max(np.median(abs(E2[g['lab_id']==l,:-1])) for l in np.unique(g['lab_id']))],4)
 for label,bs in [('linear + square',[(drift,T),(drift,Q)]),('linear + saturating',[(drift,T),(drift,S)]),('relax + linear',[(drift,R[:,0]),(drift,R[:,1]),(drift,T)]),('drift nonlinear',[(a,b) for a in (np.ones_like(drift),drift,drift**2) for b in (T,S)]),('hold + nonlinear drift',[(hold,S)]+[(a,b) for a in (np.ones_like(drift),drift,drift**2) for b in (T,S)]),('feature affine',[(a,b) for a in F.T for b in (T,S)])]:print(label,score(bs))
 if name=='diag':
  for l in np.unique(g['lab_id']):
   print('lab',l,'diag/tr/test reference centers/scales',[(round(np.median(q['ref_coupons'][q['lab_id']==l].mean(0)),3),round(np.median(q['ref_coupons'][q['lab_id']==l].std(0)),3)) for q in (g,tr,te)])
