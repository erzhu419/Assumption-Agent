from chamber_probe import *
import json
b=basis(tau); xd=m._features(g,g)[:len(h)]; dr=xd@th[2:]; L=g['lab_id']; missing=n.isnan(g['coupon_features_raw']); full=~missing.any(1); target=y-h[:,None]*(b[:,:2]@th[:2]);
# Estimate the temporal multiplier before estimating feature values in missing rows.
for scale_prior in [False,True]:
 def fit(mask):
  p=least_squares(lambda p:((dr[:,None]*(b@p[:3])+((b@p[3:]) if len(p)>3 else 0)-target)[mask]).ravel(),[.235,1.8,.03]+([0.,0.,0.] if scale_prior else []),loss='soft_l1',f_scale=.0025).x
  v=b@p[:3]; bias=b@p[3:] if scale_prior else n.zeros(len(v)); implied=(target-bias)@v/(v@v); return p,implied
 p,implied=fit(full); print('law',p.round(6)); print('implied impute',[(int(j),xd[i,j+1],xd[i,j+1]+(implied[i]-dr[i])/th[3+j]) for i,j in zip(*n.where(missing))]);
 for name,a in [('joint',X),('test',m._features(te,te)),('train',m._features(tr,tr)),('diagnostic',xd)]:
  s={'joint':{k:n.r_[tr[k],te[k]] for k in ('coupon_features_raw',)},'test':te,'train':tr,'diagnostic':g}[name]; M=n.isnan(s['coupon_features_raw']); print('impute',name,[round(a[:len(M),j+1][M[:,j]][0],5) if M[:,j].any() else None for j in range(4)])
 for i in n.unique(L):
  pp,ip=fit(full&(L!=i)); print('lab',i,'missing delta', (ip-dr)[(L==i)&~full].round(6))
for row in json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')):
 print('apparatus',[(k,v) for k,v in row.items() if k!='facility_telemetry_trace'])
print('dataset keys',list(d)); print('public item keys',list(pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]));
