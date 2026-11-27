exec(open('/tmp/m98_amplitude.py').read().split("print('rank-one")[0])
from itertools import combinations
from scipy.optimize import least_squares
np.set_printoptions(precision=5,suppress=True,linewidth=135)
# Transform diagnostics with the deployment panel frozen, rather than allowing held-out diagnostics to change it.
labs=np.r_[tr['lab_id'],te['lab_id']]; refs=np.vstack([tr['ref_coupons'],te['ref_coupons']]); xp=old._features(tr,te); rawp=np.vstack([tr['coupon_features_raw'],te['coupon_features_raw']]); imp=np.array([xp[np.isnan(rawp[:,j]),j][0] if np.isnan(rawp[:,j]).any() else np.mean(xp[:,j]) for j in range(4)]); raw=ch['coupon_features_raw']; xx=np.empty_like(raw)
for g in np.unique(ch['lab_id']):
 s=ch['lab_id']==g; rr=refs[labs==g]; xx[s]=(raw[s]-np.median(rr.mean(0)))/np.median(rr.std(0))
xx=np.where(np.isnan(xx),imp,xx); print('frozen panel delta max',np.max(abs(xx-x),axis=0)); print('test feature quantiles',np.quantile(xp[len(tr['lab_id']):],[0,.5,1],axis=0))
for tag,s in [('train',tr),('test',te)]:
 a=s['aux_chamber_telemetry']; print(tag,'telemetry quantiles',np.quantile(a,[0,.1,.5,.9,1],axis=0),'first rows',a[:4]); print('by lab means',np.array([a[s['lab_id']==g].mean(0) for g in np.unique(s['lab_id'])]))
print('archive trace summary',np.array(__import__('json').load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json'))[-1]['facility_telemetry_trace'])[:3]); print('beta',b,'kernel alpha',alpha)
for tag,x in [('joint',x),('frozen',xx)]:
 N=len(x); r=b[2]+x@b[3:]; labels=ch['lab_id']; groups=[labels==g for g in np.unique(labels)]; base=np.c_[np.ones(N),x]; lib={}
 def add(k,v):
  if np.std(v)>1e-8: lib[k]=v
 z={f'x{j}':x[:,j] for j in range(4)}; z.update({'r':r,'lin':r-b[2]})
 for i,j in combinations(range(4),2):
  for a in [-2,-1,-.5,.5,1,2]: z[f'x{i}{a:+g}x{j}']=x[:,i]+a*x[:,j]
 for k,v in z.items():
  for p in [2,3,4]: add(f'{k}^{p}',v**p)
  add(f'tanh {k}',np.tanh(v)); add(f'sinh {k}',np.sinh(v)); add(f'abs {k}',abs(v)); add(f'r*{k}',r*v)
  for a in [.1,.3,1,3]:
   add(f'{k}/(1+{a}*{k}^2)',v/(1+a*v*v)); add(f'r/(1+{a}*{k}^2)',r/(1+a*v*v))
 for ids in combinations_with_replacement(range(4),3): add(f'product {ids}',np.prod(x[:,ids],axis=1))
 for j in range(4):
  add(f'x{j} missing',np.isnan(raw[:,j])); add(f'x{j}*r^2',x[:,j]*r*r)
 def stats(A,y):
  u,s,v=np.linalg.svd(A,full_matrices=False); ok=s>s[0]*1e-8; u=u[:,ok]; p=u@(u.T@y); e=p-y; loo=e/np.maximum(1-(u*u).sum(1),1e-6); return np.linalg.norm(loo)/N**.5,np.linalg.norm(e)/N**.5,np.max(abs(loo))
 for basetag,A,y in [('fixed',np.ones((N,1)),target-r),('free',base,target)]:
  win=[]; names=list(lib)
  for i,k in enumerate(names):
   for m in [None]+names[i+1:]:
    design=np.c_[A,lib[k]] if m is None else np.c_[A,lib[k],lib[m]]; win.append((*stats(design,y),k,m))
  for row in sorted(win,key=lambda v:v[:3])[:4]:
   l,f,m,k,j=row; a=np.c_[A,lib[k]] if j is None else np.c_[A,lib[k],lib[j]]; e=np.concatenate([a[g]@np.linalg.lstsq(a[~g],y[~g],rcond=1e-8)[0]-y[g] for g in groups]); print(tag,basetag,k,j,'LOO, fit, worst, lab',np.round([l,f,m,np.sqrt(np.mean(e*e))],5),'coef',np.round(np.linalg.lstsq(a,y,rcond=1e-8)[0],5))
