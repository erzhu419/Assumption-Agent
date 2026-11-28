import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 from chamber_probe import *
from chamber_probe_base import _features
from scipy.optimize import minimize_scalar
xd=_features(q,q)[0]; good=np.isfinite(q['coupon_features_raw']).all(1); base=intrinsic(q,xd); err=obs-base; dr=augment(xd)@b[2:]
print('complete drift:',dr[good]); print('complete normalized median residual:',np.median(err[good]/dr[good,None],0));
for name,amp in [('drift',dr),('ones',np.ones(len(dr))),('hold',h),('z0',z[:,0]),('z1',z[:,1])]:
 aa=amp[good]; e=err[good]; shape=aa@e/(aa@aa); print(name,'shape',shape,'RMS',np.sqrt(np.mean((e-aa[:,None]*shape)**2)))
shape=dr[good]@err[good]/(dr[good]@dr[good]); print('intrinsic basis coef',np.linalg.lstsq(phi,shape,rcond=None)[0]); print('intrinsic basis residual',shape-phi@np.linalg.lstsq(phi,shape,rcond=None)[0]); print('incomplete residual RMS diag-own',np.sqrt(np.mean((err[~good]-dr[~good,None]*shape)**2))); print('diagnostic own imputation',xd[:][np.isnan(q['coupon_features_raw'])]);
for name,f in [('quadratic saturating',lambda v: np.r_[t*t/(1+v*t),0]),('withdrawal exponential',lambda v: np.r_[t+v*np.expm1(-t/v),0]),('squared exponential',lambda v: np.r_[np.expm1(-t/v)**2,0]),('exp minus intrinsic',lambda v: np.r_[np.expm1(-t/v)-np.expm1(-t/taus[0]),-1/v+1/taus[0]]),('quadratic gaussian',lambda v: np.r_[-np.expm1(-(t/v)**2),0])]:
 def err1(v):
  s=f(v); a=(s@shape)/(s@s); return np.linalg.norm(a*s-shape),a
 o=minimize_scalar(lambda v:err1(v)[0],bounds=(.03,20),method='bounded'); print(name,o.x,err1(o.x))
for subset in [good,~good]:
 print('amplitude ratios', (err[subset]@shape)/(shape@shape)/dr[subset]); print('projected shape RMS', np.sqrt(np.mean((err[subset] - ((err[subset]@shape)/(shape@shape))[:,None]*shape)**2)))
