import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 import chamber_probe as p
import numpy as np
from scipy.optimize import least_squares
m,g,t,Y,R,theta=p.m,p.g,p.t,p.Y,p.R,p.theta
x=m._features(g,g)[0]; _,h,F=m._specimen(g,x,p.bridge); E=Y-(h[:,None]*theta[:2])@R.T
# Fit only physically differentiable time kernels, preserving the observed rate constraint.
for name, fn in {
'exp':lambda z:-np.expm1(-t*z)/z,
'log':lambda z:np.log1p(t*z)/z,
'rat':lambda z:t/(1+t*z),
'rat2':lambda z:t/(1+(t*z)**2),
'pow':lambda z:t*(1+t*z)**(-.5),
}.items():
 def pred(z): return z[:,0,None]*fn(z[:,1,None])
 z=least_squares(lambda v:np.r_[(pred(v.reshape(-1,2))-E[:,:-1]).ravel(),(v[::2]-E[:,-1])],np.column_stack((E[:,-1],np.full(len(h),.3))).ravel(),bounds=(np.tile([-3,.00001],len(h)),np.tile([3,10],len(h)))).x.reshape(-1,2)
 print(name,'rmse',np.sqrt(np.mean((pred(z)-E[:,:-1])**2)),np.sqrt(np.mean((z[:,0]-E[:,-1])**2)))
 if name in ('exp','log','rat'): print(np.round(np.column_stack((F@theta[2:],h,z)),3).tolist())
# A low-rank temporal basis reveals which observed inputs drive its amplitudes.
for Aname,A in [('raw',g['coupon_features_raw']),('diag',x),('joint',m._features(p.joined([p.tr,p.te]),g)[1])]:
 a=np.column_stack((np.ones(len(h)),A,h)); ok=np.isfinite(a).all(1); co=np.linalg.lstsq(a[ok],E[ok],rcond=None)[0]; print(Aname,'OLS residual',np.round(np.sqrt(np.mean((a[ok]@co-E[ok])**2,axis=0))[[0,5,11,12]],4),'coefs',np.round(co[:,[0,5,11,12]],3).tolist())
