from chamber_probe import *
from scipy.optimize import minimize
print('tau',tau,'weights',w)
print('correction SVD',np.linalg.svd(y-pred,compute_uv=False))
for j in [0,2,5,9,11,12]:
 v=(y-pred)[:,j]; print('residual t index',j,'correlations hold/history/state/drift/x',np.round(np.corrcoef(np.c_[v,h,past,z,r,x].T)[0,1:],3))
for name,f in [('hold,z,drift',np.c_[h,z,r]),('hold,z,drift,1',np.c_[h,z,r,ones]),('hold,past,drift,1',np.c_[h,past,r,ones]),('hold,past,x,1',np.c_[h,past,x,ones])]:
 c,e=profile(f,y); print(name,'fixed',np.round(stats(e),5),'coeff',np.round(c,4).tolist())
 def obj(logt):
  c,e=profile(f,y,np.exp(logt)); return np.log1p((e/.008)**2).sum()
 m=minimize(obj,np.log(tau),method='Nelder-Mead',bounds=list(zip(np.log([.3,1.5]),np.log([1.2,12]))),options={'maxiter':100}); t=np.exp(m.x); c,e=profile(f,y,t)
 print('free',np.round(t,4),np.round(stats(e),5),'coeff',np.round(c,4).tolist())
print('observed vs baseline residual by specimen')
print(np.round(np.c_[g['lab_id'],g['stratum'],h,z,r,x,(y-pred)[:,[0,3,7,11,12]]],4))
# A flexible per-coupon fit isolates the time dependence from the covariate law.
B=basis(tau); C=np.linalg.lstsq(B,y.T,rcond=None)[0].T
print('individual temporal basis errors',stats(C@B.T-y)); print('amplitude feature correlations',np.round(np.corrcoef(np.c_[C,h,z,r,x].T)[:3,3:],3))
