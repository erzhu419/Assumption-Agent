exec(open('/tmp/chamber_probe.py').read().split("for name,xx")[0])
from itertools import product

def features(s, balanced):
 lab=s['lab_id']; st=s['stratum']; ref=s['ref_coupons']; z=s['coupon_features_raw'].copy()
 for label in np.unique(lab):
  m=lab==label; v=ref[m]; z[m]=(z[m]-np.median(v.mean(0)))/np.median(v.std(0))
 if balanced:
  _,ix,n=np.unique(np.column_stack((lab,st)),axis=0,return_counts=True,return_inverse=True); w=1/n[ix]
 else: w=np.ones(len(z))
 for c in z.T:
  m=np.isfinite(c); c[~m]=np.average(c[m],weights=w[m])
 return z

H=h[:,None]*theta[:2]; base=H@relax.T; rate0=H@(1/tau)
missing=np.isnan(q['coupon_features_raw']).any(1)
for bal in (False,True):
 u=theta[2]+features(q,bal)@theta[3:]
 def err(v):
  f,s=v; k=f*tau[0]*relax[:,0]-(1-f)*s*np.expm1(-t/s); return base+u[:,None]*k-y
 def fit(m):
  return least_squares(lambda v:err(v)[m].ravel(),[.25,8],bounds=([0,.05],[1,50]),loss='soft_l1',f_scale=.004).x
 print('balanced',bal,'rate abs p90 all/missing/complete',[np.quantile(abs(rate0+u-r)[m],.9) for m in (np.ones(len(u),bool),missing,~missing)])
 oof=np.empty_like(y); vv=[]
 for i in range(len(y)):
  v=fit(np.arange(len(y))!=i); vv.append(v); oof[i]=err(v)[i]
 print('full fit',fit(np.ones(len(y),bool)),'LOO parameters min/max',np.min(vv,0),np.max(vv,0))
 for name,e in [('LOO',oof),('0.25/8',err([.25,8])),('0.27/8',err([.27,8]))]:
  print(name,'curve abs p90 all/missing/complete/late',[np.quantile(abs(w),.9) for w in (e,e[missing],e[~missing],e[:,-3:])])
