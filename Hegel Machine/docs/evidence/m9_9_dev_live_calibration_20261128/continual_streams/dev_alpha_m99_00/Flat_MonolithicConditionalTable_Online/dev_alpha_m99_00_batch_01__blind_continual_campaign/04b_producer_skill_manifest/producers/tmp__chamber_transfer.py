import contextlib,io,runpy
import numpy as np
from scipy.optimize import least_squares
with contextlib.redirect_stdout(io.StringIO()): p=runpy.run_path('/tmp/chamber_probe.py')
globals().update({k:p[k] for k in ('m','d','tr','te','di','dt','t','tau','theta','bridge','n','h','v','vr')})
O=np.c_[v,vr]; W=np.r_[np.ones(len(t)),2.]; E=lambda a:np.r_[-np.expm1(-t/a),1/a]; L=np.r_[t,1.]; reshape=lambda a:np.asarray(a).reshape(O.shape)
for label,x in [('diag',m._features(di)),('all',m._features(tr,te,di)[n+h:])]:
 past,hold,_=m._coupon(di,x,bridge); f=x@theta[3:];
 A=np.c_[np.ones(len(f)),hold,f]; C=np.linalg.lstsq(A,O,rcond=None)[0]; sig=theta[0]*E(tau[0])+theta[1]*E(tau[1]);
 print(label,'design residual RMS',np.sqrt(np.mean((A@C-O)**2,axis=0)).round(4)); print('hold transfer vs time and derivative:',C[1].round(5)); print('hold prior same:',sig.round(5)); print('feature transfer:',C[2].round(5)); print('intercept:',C[0].round(5)); print('hold minus baseline:',(C[1]-sig).round(5));
 def fit_hold(z): return (z[0]*E(np.exp(z[2]))+z[1]*E(np.exp(z[3]))-C[1])*W
 q=least_squares(fit_hold,[*theta[:2],*np.log(tau)]).x; print('hold-only two-exponential weights/taus',q[:2],np.exp(q[2:]),'RMS',np.sqrt(np.mean(fit_hold(q)**2)))
 for name,ta in [('intrinsic',tau),('effective',np.array([.50,2.52]))]:
  B=np.stack([E(ta[0]),E(ta[1]),L]); print(name,'coeff matrix',np.linalg.lstsq((B*W).T,(C*W).T,rcond=None)[0].T.round(5));
 # Independently estimate hold and feature response by variable projection, testing shared/fixed and extra-branch bases.
 for form in ('shared','separate','hold-fixed','single-feature','fixed-reset'):
  def design(z,j):
   a=np.exp(z)
   if form=='shared': return np.stack([E(a[0]),E(a[1]),L])
   if form=='separate': return np.stack([E(a[0]),E(a[1]),L]) if j==1 else np.stack([E(a[2]),E(a[3]),L])
   if form=='hold-fixed': return sig[None,:] if j==1 else np.stack([E(a[0]),E(a[1]),L])
   if form=='single-feature': return np.stack([E(a[0]),E(a[1]),L]) if j==1 else np.stack([E(a[2]),L])
   return sig[None,:] if j==1 else np.stack([E(tau[0]),E(tau[1]),L])
  def calc(z,C=C):
   vals=[]; coeff=[]
   for j in range(3):
    B=design(z,j); w=np.linalg.lstsq((B*W).T,C[j]*W,rcond=None)[0]; vals.append(w@B); coeff.append(w)
   return np.array(vals),[x.round(5).tolist() for x in coeff]
  ini={'shared':[.5,2.5],'separate':[.5,4,.5,2.5],'hold-fixed':[.5,2.5],'single-feature':[.5,4,1.5],'fixed-reset':[]}[form]; z=np.log(ini)
  if len(z): z=least_squares(lambda z:((A@calc(z)[0]-O)*W).ravel(),z,bounds=(np.log(.05),np.log(25)),max_nfev=140).x
  P,Q=calc(z); r=A@P-O; print(form,'taus',np.exp(z).round(4),'coeff',Q,'curve/rate RMS',round(np.sqrt(np.mean(r[:,:-1]**2)),6),round(np.sqrt(np.mean(r[:,-1]**2)),6))
 print('diag range hold',np.quantile(hold,[0,.5,1]).round(3),'past std',past.std(0).round(3),'x range',np.stack([x.min(0),x.max(0)]).round(3))
