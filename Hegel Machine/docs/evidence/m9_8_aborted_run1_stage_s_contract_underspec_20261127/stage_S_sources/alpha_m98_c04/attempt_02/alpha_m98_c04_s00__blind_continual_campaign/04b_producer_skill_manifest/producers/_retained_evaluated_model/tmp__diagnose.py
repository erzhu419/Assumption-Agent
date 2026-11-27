from explore import *
from collections import defaultdict

def take(g, m): return {k:v[m] for k,v in g.items()}
def check(key):
 g=D[key]; print('\n',key); v=fit(g); e=basis(g,np.exp(v[:2]))@v[2:]-target(g)
 print('rate/curve residual covariance',np.corrcoef(e.T)[-1,:].round(2)); print('residual quantiles by horizon',np.quantile(abs(e),[.5,.8,.95],axis=0).round(4))
 for j in np.unique(g['lab_id']):
  q=g['lab_id']==j; r=e[q]; print('lab',j,'n',q.sum(),'missing',np.isnan(g['coupon_features_raw'][q]).sum(),'rate, early/mid/late signed means',np.mean(r[:,[-1,0,5,11]],axis=0).round(4))
 m=np.isnan(g['coupon_features_raw']).any(1)
 for q in [m,~m]:
  if q.any(): print('missing slice',q.sum(),'median abs curve/rate',np.median(abs(e[q,:-1])),np.median(abs(e[q,-1])))
 return v
vt=check('train'); vp=check('chamber_verification_trials'); g=D['chamber_verification_trials']; e=basis(g,np.exp(vt[:2]))@vt[2:]-target(g)
print('\nbenchtop to chamber lab error (curve/rate MAE)')
for j in np.unique(g['lab_id']):
 q=g['lab_id']==j; print(j,np.mean(abs(e[q,:-1])),np.mean(abs(e[q,-1])))
for key in ['train','test','chamber_verification_trials']:
 g=D[key]; print('\npanel',key)
 for j in np.unique(g['lab_id']):
  q=g['lab_id']==j; r=g['ref_coupons'][q]; print(j,q.sum(),round(np.median(r.mean(0)),4),round(np.median(r.std(0)),4))
