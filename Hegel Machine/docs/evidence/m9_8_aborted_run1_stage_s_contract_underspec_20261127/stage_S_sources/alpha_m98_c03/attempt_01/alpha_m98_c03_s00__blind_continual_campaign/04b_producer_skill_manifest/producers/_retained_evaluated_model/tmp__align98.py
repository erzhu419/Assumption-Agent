from profile98 import *
from itertools import combinations
bench,pilot,test=S
sizes=[len(s['lab_id']) for s in S]
keys='lab_id stratum ref_coupons coupon_features_raw'.split()
preps={}
for n in (1,2,3):
 for group in combinations(range(3),n):
  a=combine([S[k] for k in group]); _,st,im=prep(a); preps[group]=(a,st,im)
for k,s in enumerate(S[:2]):
 tau,w,drift,e=fit(s); miss=np.isnan(s['coupon_features_raw']); complete=~miss.any(1); print('\nsource',k,'tau',tau,'w',w)
 for group,(_,st,im) in preps.items():
  x=prep(s,st,preps[(0,)][2])[0]; A=np.c_[np.ones(len(x)),x]; q=least_squares(lambda p:A[complete]@p-drift[complete],np.linalg.lstsq(A[complete],drift[complete],rcond=None)[0],loss='soft_l1',f_scale=.006); oof=np.empty(len(x))
  for lab in np.unique(s['lab_id']):
   keep=(s['lab_id']!=lab)&complete; a=A[keep]; p=least_squares(lambda p:a@p-drift[keep],q.x,loss='soft_l1',f_scale=.006).x; oof[s['lab_id']==lab]=A[s['lab_id']==lab]@p-drift[s['lab_id']==lab]
  print(group, q.x.round(5),'complete fitted RMS',np.sqrt(np.mean(q.fun**2)).round(5),'worst lab oof',max(np.mean(abs(oof[(s['lab_id']==lab)&complete])) for lab in np.unique(s['lab_id']) if np.any((s['lab_id']==lab)&complete)).round(5))
  if group in ((0,2),(1,)):
   er=drift-A@q.x; implied=(er[:,None]/q.x[1:]+x)[miss]; print(' inferred missing fillers (lab,stratum,feature,implied)', np.c_[s['lab_id'][miss.any(1)],s['stratum'][miss.any(1)],np.where(miss)[1],implied].round(4)); print(' supplied possible fillers',im.round(4),preps[(0,)][2].round(4),preps[(0,2)][2].round(4)); print(' panel',st)
 print('resid vs time MAE',np.mean(abs(e),axis=0).round(4))
