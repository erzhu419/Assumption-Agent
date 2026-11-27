from rheology_probe import *
from itertools import combinations
for label,s in [('bench',TR),('pilot',P)]:
 n=len(s['lab_id']); labs=s['lab_id']; good=np.isfinite(s['coupon_features_raw']).all(1); eye=np.eye(n); sy=.005 if label=='pilot' else .012; sr=sy
 y=s['strain_curves_obs']-s['strain_0'][:,None]; r=s['instant_rate_obs']; z=np.r_[y.ravel(),r]
 def mat(t):
  A,R=basis(s,eye,t); return np.vstack([A.reshape(-1,n+2),R])
 def residual(v):return (mat(v[:2])@v[2:]-z)/sy
 t=[.55,4.5]; fit1=least_squares(residual,np.r_[t,np.linalg.lstsq(mat(t),z,rcond=None)[0]],bounds=(np.r_[[.050001]*2,[-np.inf]*(n+2)],np.r_[[24.99999]*2,[np.inf]*(n+2)]),loss='cauchy',x_scale='jac'); drift=fit1.x[4:]
 print(label,'spectrum',fit1.x[:4].round(6),'complete',int(good.sum()))
 variants={'RT':[TR,TE],'RTP':[TR,TE,P],'TP':[TE,P],'R':[TR],'T':[TE]}
 for name,refs in variants.items():
  x=features(s,refs,TR); j=np.flatnonzero(good); e=[]; bs=np.linalg.lstsq(x[j],drift[j],rcond=None)[0]
  for i in j:
   q=j[j!=i]; e.append(x[i]@np.linalg.lstsq(x[q],drift[q],rcond=None)[0]-drift[i]);
  print(name,'beta',np.round(bs,4),'complete rms/LOO rms/max',*[round(a,5) for a in (np.sqrt(np.mean((x[j]@bs-drift[j])**2)),np.sqrt(np.mean(np.array(e)**2)),max(abs(np.array(e))) )], 'missing',np.round((x@bs-drift)[~good],4))
 if label=='pilot':
  x=features(s,[TR,TE],TR); print('pilot [lab,stratum,missing mask,x,drift]');print(np.c_[labs,s['stratum'],(~np.isfinite(s['coupon_features_raw']))@(2**np.arange(4)),x[:,1:],drift].round(4));
  avg=lambda a: np.median(a,axis=0)
  print('bench/test/pilot ref mean & spread per lab')
  for lab in np.unique(labs):
   print(lab,[(float(np.median(q.mean(0))),float(np.median(q.std(0)))) for h in (TR,TE,P) if len(q:=h['ref_coupons'][h['lab_id']==lab])])
