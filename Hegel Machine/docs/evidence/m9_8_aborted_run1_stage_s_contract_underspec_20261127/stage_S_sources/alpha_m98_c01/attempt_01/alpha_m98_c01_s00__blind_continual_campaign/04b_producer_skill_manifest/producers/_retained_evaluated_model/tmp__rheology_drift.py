from rheology_probe import *
from itertools import combinations
for label,s in [('train',TR),('pilot',P)]:
 n=len(s['lab_id']); sy=.011; sr=.011
 def fun(v):
  A,R=basis(s,np.eye(n),v[:2]); return np.r_[((A@v[2:]+s['strain_0'][:,None]-s['strain_curves_obs'])/sy).ravel(),(R@v[2:]-s['instant_rate_obs'])/sr]
 fit0,_=fit(s,features(s,[TR,TE,P],TR)); tau=fit0[:2]
 A,R=basis(s,np.eye(n),tau); rhs=np.r_[(s['strain_curves_obs']-s['strain_0'][:,None]).ravel(),s['instant_rate_obs']]; B=np.vstack([A.reshape(-1,n+2),R]); init=np.linalg.lstsq(B,rhs,rcond=None)[0]; npar=n+2
 sol=least_squares(fun,np.r_[tau,init],bounds=(np.r_[[.050001]*2,[-np.inf]*npar],np.r_[[24.99999]*2,[np.inf]*npar]),loss='cauchy',x_scale='jac',max_nfev=250)
 e=fun(sol.x); drift=sol.x[4:]; print(label,'taus and weights',sol.x[:4].round(5),'curve/rate median abs',np.median(abs(e[:-n]))*sy,np.median(abs(e[-n:]))*sr)
 sets={'R':TR,'T':TE,'P':P}; results=[]
 for size in (1,2,3):
  for ks in combinations(sets,size):
   refs=[sets[k] for k in ks]; supported=np.unique(np.concatenate([r['lab_id'] for r in refs])); val=np.isin(s['lab_id'],supported)
   if not val.all(): continue
   for fs in ('R','T','P','RT','RP','RTP'):
    fill={k:np.concatenate([sets[j][k] for j in fs]) for k in ['lab_id','stratum','coupon_features_raw']}
    if not np.isin(fill['lab_id'],supported).all():continue
    x=features(s,refs,fill)
    beta=np.linalg.lstsq(x,drift,rcond=None)[0]; ff=least_squares(lambda b:(x@b-drift)/.008,beta,loss='cauchy',max_nfev=100); err=x@ff.x-drift
    results.append((np.median(abs(err)),np.mean(np.log1p((err/.008)**2)),''.join(ks),fs,ff.x,err))
 for mae,loss,refs,fill,beta,err in sorted(results,key=lambda r:r[1])[:6]:
  print('refs/fill',refs,fill,'med',round(mae,5),'loss',round(loss,4),'max',round(np.max(abs(err)),4),'beta',beta.round(4))
  if label=='pilot': print('  residual by lab',[(int(j),np.round(err[s['lab_id']==j],3).tolist()) for j in np.unique(s['lab_id'])])
 if label=='pilot':
  z=(e[:-n]*sy).reshape(n,-1); print('pilot sample residuals [lab, #missing, drift, median abs curve err, rate err, # curve err > .05]');print(np.c_[s['lab_id'],np.isnan(s['coupon_features_raw']).sum(1),drift,np.median(abs(z),axis=1),e[-n:]*sr,(abs(z)>.05).sum(1)].round(4))
