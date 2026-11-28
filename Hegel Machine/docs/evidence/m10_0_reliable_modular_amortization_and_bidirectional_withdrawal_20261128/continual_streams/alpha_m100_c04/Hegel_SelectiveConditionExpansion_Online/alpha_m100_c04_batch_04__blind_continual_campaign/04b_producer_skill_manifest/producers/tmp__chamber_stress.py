exec(open('/tmp/chamber_probe.py').read().split('for tt in (ta,')[0])
import downstream_solver as s
from batch_solver import solve_batch
from itertools import combinations
from scipy.optimize import least_squares
lab=d['lab_id']; raw=d['coupon_features_raw']; panel=d['ref_coupons']; xx=np.empty_like(raw)
for l in np.unique(lab):
 m=lab==l; pp=panel[m]; xx[m]=(raw[m]-np.median(pp.mean(0)))/np.median(pp.std(0,ddof=0))
full=np.isfinite(xx).all(1); pulse=np.c_[past, h]; tbase=F(ta); B=h[:,None]*(tbase[:,:2]@th[:2]); R=Y-B
z=h[:,None]-design[1][:,:2]*ta
# Fit both response channels jointly; select optional terms strictly within the outer training labs.
def fit(Q,mask):
 A=(Q[:,None,:,None]*tbase[None,:,None,:]).reshape(len(h),len(t)+1,-1)
 c=np.linalg.lstsq(A[mask].reshape(-1,A.shape[-1]),R[mask].ravel(),rcond=None)[0]
 return B+A@c
maxima={}
for out in np.unique(lab):
 train=lab!=out; holdout=~train
 # Reference coordinates use each panel, not its labels. Compare the diagnostic-only
 # imputation indicated by measurements against required deployment imputation.
 v=th[2]+np.where(np.isfinite(xx),xx,np.nanmean(xx[train],axis=0))@th[3:]
 vb=np.c_[np.ones(len(h)),s._features(d,{k:a[:0] for k,a in d.items()})]@th[2:]
 gs={'rank_one':v[:,None], 'baseline_diag_imputation':vb[:,None]}
 terms={'signed-memory':[z[:,0],z[:,1],z[:,0]-z[:,1],h-z[:,0],h-z[:,1]], 'pairwise-pulses':[pulse[:,i]-pulse[:,j] for i,j in combinations(range(pulse.shape[1]),2)], 'product':[h*v,z[:,0]*z[:,1],h*z[:,0]*v,v*v], 'rational':[v/(1+a*q*q)-v for q in (h,z[:,0],z[:,1],z[:,0]-z[:,1],v) for a in (.03,.3,3.)]}
 for fam,candidates in terms.items():
  mats=[np.c_[v,q] for q in candidates]
  def score(Q):
   es=[]
   for inner in np.unique(lab[train]):
    m=lab==inner; E=fit(Q,train&~m&full)[m&full]-Y[m&full]; es.append(max(np.sqrt(np.mean(E**2,axis=0))))
   return max(es)
  gs[fam]=min(mats,key=score)
 common=fit(gs['rank_one'],train&full)-Y
 for name,Q in gs.items():
  E=fit(Q,train&full)-Y
  for slice_,m in [('complete',holdout&full),('all',holdout),('imputed',holdout&~full)]:
   if m.any():
    pair=np.array([np.max(np.sqrt(np.mean(E[m,:-1]**2,axis=0))),np.sqrt(np.mean(E[m,-1]**2))]); key=(name,slice_); maxima[key]=np.maximum(maxima.get(key,0),pair)
 # Within-lab drift sensitivity and residual bias at the longest time and at t=0.
 print('lab',out,'complete centered drift/response r',round(np.corrcoef(v[holdout&full],R[holdout&full,-2])[0,1],5),'mean strain/rate residual',np.round(common[holdout&full,-2:].mean(0),5))
for key,val in maxima.items(): print(key,np.round(val,6))
ans=solve_batch(D)
assert all(np.isfinite(a).all() for a in ans.values())
assert .05<ans['estimated_taus'][0]<ans['estimated_taus'][1]<25
print('deployment rate range',np.min(ans['initial_rate_pred']),np.max(ans['initial_rate_pred']))
