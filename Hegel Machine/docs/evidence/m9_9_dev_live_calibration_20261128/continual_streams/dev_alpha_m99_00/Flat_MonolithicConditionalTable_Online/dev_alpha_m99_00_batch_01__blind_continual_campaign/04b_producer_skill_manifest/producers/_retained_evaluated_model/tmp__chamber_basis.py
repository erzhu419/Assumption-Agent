import runpy, numpy as np
from scipy.optimize import least_squares
p=runpy.run_path('/tmp/chamber_probe.py'); globals().update({k:p[k] for k in ('m','d','tr','te','di','dt','t','tau','theta','bridge','n','h','v','vr')})
for label,xd in [('diagnostics-only',m._features(di)),('all-three',m._features(tr,te,di)[n+h:])]:
 past,hold,dr=m._coupon(di,xd,bridge); raw=np.c_[np.ones(len(hold)),hold,past,xd]; names=['const','hold']+['past'+str(j) for j in range(6)]+['x'+str(j) for j in range(4)]; obs=np.c_[v,vr]
 def basis(ta): return np.vstack((np.c_[-np.expm1(-t[:,None]/ta),t],np.r_[1/ta,1]))
 def profile(ta):
  B=basis(ta); w=np.r_[np.ones(len(t)),2]; coef=np.linalg.lstsq(w[:,None]*B, (obs*w).T,rcond=None)[0].T; return coef, (coef@B.T-obs)*w
 fit=least_squares(lambda z:profile(np.exp(z))[1].ravel(),np.log(tau),bounds=(np.log(.05001),np.log(24.99)))
 print(label,'free taus',np.exp(fit.x),'RMS fixed/free',*[np.sqrt(np.mean(profile(z)[1]**2)) for z in (tau,np.exp(fit.x))]);
 for ta in (tau,np.exp(fit.x)):
  A,E=profile(ta); reg=np.linalg.lstsq(raw,A,rcond=None)[0]; print('coef fit RMS',np.sqrt(np.mean((raw@reg-A)**2,axis=0))); print(np.round(reg.T,4)); print('dominant coupon amplitudes',np.round(A[:5],4))
 print('columns',names)
