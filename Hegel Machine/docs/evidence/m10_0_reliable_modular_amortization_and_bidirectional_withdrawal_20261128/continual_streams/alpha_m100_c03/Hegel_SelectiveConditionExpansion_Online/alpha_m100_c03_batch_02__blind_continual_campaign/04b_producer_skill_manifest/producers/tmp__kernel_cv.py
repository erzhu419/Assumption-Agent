from chamber_probe import *
from downstream_solver import _features
from scipy.optimize import minimize_scalar
from scipy.special import erf
tr,te=d['train'],d['test']; labs=np.asarray(g['lab_id']); F=y-basis(tau)[:,:2]@(w[:2]*gap).T if False else y-(w[:2]*gap)@basis(tau)[:,:2].T
kernels={'exp':lambda k: np.r_[-np.expm1(-k*times)/k,1.], 'atan':lambda k:np.r_[np.arctan(k*times)/k,1.], 'erf':lambda k:np.r_[np.sqrt(np.pi)*erf(k*times)/(2*k),1.], 'rational':lambda k:np.r_[times/(1+k*times),1.], 'linear':lambda k:np.r_[times,1.]}
xg=_features(g,g)[-len(h):]; rg=w[2]+xg@w[3:]
for name,fun in kernels.items():
 def fit(m):
  def obj(k):
   v=fun(k); a=F[m]@v/(v@v); return np.sum((F[m]-a[:,None]*v)**2)
  return minimize_scalar(obj,bounds=(.1,4.),method='bounded').x
 k=fit(np.ones(len(h),bool)); fitv=fun(k); amp=F@fitv/(fitv@fitv); print(name,'full k',round(k,6),'profile',np.round(stats(amp[:,None]*fitv-F),5),'coef amplitude vs intrinsic',np.round(np.linalg.lstsq(np.c_[ones,rg],amp,rcond=None)[0],6)); print('  lab CV k & RMSE',[(int(l),round(kl,6),np.round(stats(rg[m,None]*fun(kl)-F[m])[:2],5).tolist()) for l in np.unique(labs) for m in [labs==l] for kl in [fit(~m)]])
print('eval times',times,'splits',[(name,len(s['lab_id']),list(s)) for name,s in [('train',tr),('test',te),('diag',g)]]); print('diagnostic missing',np.isnan(g['coupon_features_raw']).sum(axis=0),'test missing',np.isnan(te['coupon_features_raw']).sum(axis=0)); print('lab reference dispersion',[(name,[(int(l),round(np.median(s['ref_coupons'][s['lab_id']==l].mean(0)),3),round(np.median(s['ref_coupons'][s['lab_id']==l].std(0)),3)) for l in np.unique(s['lab_id'])]) for name,s in [('train',tr),('test',te),('diag',g)]])
