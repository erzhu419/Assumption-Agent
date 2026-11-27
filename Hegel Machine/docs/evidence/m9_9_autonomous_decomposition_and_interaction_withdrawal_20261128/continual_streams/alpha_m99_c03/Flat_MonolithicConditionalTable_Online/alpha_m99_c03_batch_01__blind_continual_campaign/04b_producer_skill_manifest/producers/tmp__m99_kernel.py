from m99_inputs import *
from scipy.optimize import least_squares
lab=g['lab_id']; st=g['stratum']; X=m._features(g,g)[-len(lab):]; a=th[2]+X@th[3:]; p,h,_=cg
T=np.r_[t,0.]; D=np.vstack((E,er)); obs=np.c_[y,yr]; hold=h[:,None]*(D[:,:2]@th[:2]); R=obs-hold
k=(a@R)/(a@a); print('times',t.round(4)); print('empirical kernel',k.round(5)); print('kernel residual by horizon',np.sqrt(np.mean((R-a[:,None]*k)**2,0)).round(5)); print('time slopes relative to hold', np.linalg.lstsq(np.c_[h,a],obs,rcond=None)[0][0].round(5)); print('nominal hold kernels',(D[:,:2]@th[:2]).round(5))
def K(q): return np.r_[q*-np.expm1(-t/q),1.]
def K2(q): return np.r_[t/(1+t/q),1.]
def K3(q): return np.r_[q*np.log1p(t/q),1.]
def mix(b): return D@(b*np.r_[tau,1.])
def fmt(x): return np.round(x,6).tolist()
models=[('single exp',lambda v:K(np.exp(v[0])),[0.]),('rational',lambda v:K2(np.exp(v[0])),[0.]),('log',lambda v:K3(np.exp(v[0])),[0.]),('exp+linear',lambda v:v[1]*K(np.exp(v[0]))+(1-v[1])*T,[0.,.8]),('two exp',lambda v:v[2]*K(np.exp(v[0]))+(1-v[2])*K(np.exp(v[1])),[-.5,1.2,.5]),('shared tau normalized',lambda v:mix(np.r_[v,1.-sum(v)]),[.5,.45]),('shared tau unconstrained',mix,[.5,.45,.05])]
for name,fn,start in models:
 sol=least_squares(lambda v:(R-a[:,None]*fn(v)).ravel(),start); worst=[]
 for gr in (lab,st,np.arange(len(lab)),np.isnan(g['coupon_features_raw']).any(1)):
  err=np.zeros_like(R)
  for j in np.unique(gr):
   keep=gr!=j; s=least_squares(lambda v:(R[keep]-a[keep,None]*fn(v)).ravel(),start); err[~keep]=R[~keep]-a[~keep,None]*fn(s.x)
  worst.append(np.max([[np.mean(err[gr==j,:-1]**2)**.5,np.mean(err[gr==j,-1]**2)**.5] for j in np.unique(gr)],0))
 print(name,'param',fmt(sol.x),'kernel_delta',fmt(fn(sol.x)-k),'worst',np.round(worst,5).tolist())
