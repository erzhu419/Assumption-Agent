exec(open('/tmp/chamber_probe.py').read().split('for name,y')[0])
from scipy.linalg import lstsq
n=len(hold); lab=v['lab_id']; A=basis(tau); tail=np.r_[np.cumsum(dt[:0:-1])[::-1],0.]; q=-np.expm1(-dt[:,None]/tau)*np.exp(-tail[:,None]/tau); z=past@q
base=(hold[:,None]*w[:2])@A[:,:2].T
for label, cx in [('diag+self',m._features(v,v)[:n]),('diag+test',x),('diag+train',m._features(v,d['train'])[:n]),('all',m._features(v,{k:np.concatenate((d['train'][k],d['test'][k])) for k in ('lab_id','stratum','ref_coupons','coupon_features_raw')})[:n])]:
 dr=w[2]+cx@w[3:]; print(label,'missing rows',np.flatnonzero(np.isnan(v['coupon_features_raw']).any(1)))
 col=lambda f,b: np.asarray(f)[:,None]*A[:,b]
 variants={
 'drift plus free states':[col(z[:,b],b) for b in range(2)]+[col(dr,b) for b in range(3)],
 'drift plus state offsets':[col(z[:,b],b) for b in range(2)]+[col(dr,b) for b in range(3)]+[col(np.ones(n),b) for b in range(3)],
 'reweighted drift plus free states':[col(z[:,b],b) for b in range(2)]+[col(f,b) for b in range(3) for f in np.c_[np.ones(n),cx].T],
 'three independent memory and drift coefficients':[col(dr,b) for b in range(3)],
 'intrinsic dr plus common memory factor':[sum(col(w[b]*z[:,b],b) for b in range(2))]+[col(dr,b) for b in range(3)],
 'integrated drift drive':[sum(col(w[b]*z[:,b],b) for b in range(2)),sum(col(w[b]*tau[b]*dr,b) for b in range(2)),col(dr,2)],
 }
 for key,cols in variants.items():
  B=np.stack(cols,axis=2); target=Y-base
  def est(j): return lstsq(B[j].reshape(-1,len(cols)),target[j].ravel())[0]
  full=est(np.arange(n)); err=target-B@full; pred=np.empty_like(Y)
  for L in np.unique(lab):
   j=lab!=L; pred[~j]=base[~j]+B[~j]@est(j)
  e=Y-pred; stats=np.array([[np.mean(e[lab==L,:-1]**2)**.5,np.mean(e[lab==L,-1]**2)**.5] for L in np.unique(lab)]);print(key,'coef',np.round(full,4),'in',np.round((err**2).mean(0)**.5,4),'worst',stats.max(0));
  if key=='drift plus free states': print('coupon RMSE',np.round(np.mean(err**2,1)**.5,3))
print('state quantiles',np.quantile(z,[0,.5,1],axis=0)); print('tau*weights',tau*w[:2]); print('raw covariates summary for feature proxy');
for p in (d['train'],d['test'],v):
 for L in np.unique(p['lab_id']):
  j=p['lab_id']==L; a=p['ref_coupons'][j]; print(L,len(a),np.median(a.mean(0)),np.median(a.std(0)),np.isnan(p['coupon_features_raw'][j]).sum())
