from chamber_probe import *
from itertools import combinations
b=basis(tau); target=y-h[:,None]*(b[:,:2]@th[:2]); lab=g['lab_id']; missing=n.isnan(g['coupon_features_raw']); full=~missing.any(1)
pack=lambda *groups:{k:n.concatenate([s[k] for s in groups],axis=0) for k in ('lab_id','stratum','coupon_features_raw','ref_coupons')}
for name,a in [('diag',m._features(g,g)[-len(h):]),('diag+test',m._features(te,g)[-len(h):]),('diag+train',m._features(tr,g)[-len(h):]),('all',x)]:
 dr=a@th[2:]; F=n.c_[dr,n.ones(len(dr))]; A=(b[None,:,:,None]*F[:,None,None,:]).reshape(y.size,6)
 for k in (1,2):
  M=A[:,::2] if k==1 else A
  def pred(mask):
   ix=n.repeat(mask,y.shape[1]); Q=M[ix]; yy=target.ravel()[ix]; q=least_squares(lambda p:Q@p-yy,n.linalg.lstsq(Q,yy,rcond=None)[0],jac=lambda _:Q,loss='soft_l1',f_scale=.0025).x;return q,(M@q).reshape(y.shape)-target
  q,e=pred(full); ep=n.empty_like(e)
  for l in n.unique(lab): ep[lab==l]=pred(full&(lab!=l))[1][lab==l]
  print(name,k,'q',q.round(5),'complete pooled',n.sqrt(n.mean(ep[full]**2,0))[[0,5,-2,-1]].round(5),'complete worst lab',n.max([n.sqrt(n.mean(ep[full&(lab==i)]**2,0))[[0,5,-2,-1]] for i in n.unique(lab)],0).round(5),'missing',n.sqrt(n.mean(ep[~full]**2,0))[[0,5,-2,-1]].round(5))
  if k==1 and name=='diag':
   v=b@q; implied=target@v/(v@v); print('missing implied drift delta',n.c_[lab[~full],missing[~full],(implied-dr)[~full]].round(5)); print('rank-one orthogonal RMS',n.sqrt(n.mean((target-implied[:,None]*v)**2,0)).round(5))
for s in (tr,te,g):
 print('aux',n.shape(s.get('aux_chamber_telemetry')),n.round(s['aux_chamber_telemetry'][:3],3))
