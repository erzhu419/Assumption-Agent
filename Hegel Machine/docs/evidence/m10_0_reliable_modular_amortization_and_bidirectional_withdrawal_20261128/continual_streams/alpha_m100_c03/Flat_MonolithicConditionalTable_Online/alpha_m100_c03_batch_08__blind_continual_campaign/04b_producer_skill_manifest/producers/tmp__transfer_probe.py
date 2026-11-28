exec(open('/tmp/chamber_probe.py').read().split('for drift in')[0])
labs=d['lab_id']; E=np.r_[t,1.]; Z=h[:,None]-c[:,0,:2]/(-np.expm1(-t[0]/tau)); F0=np.column_stack([h, np.ones(nd)])
def cv(F, q=tau, ridge=0):
 A=np.vstack([et(a) for a in q]+[E]); pred=np.empty_like(y); co=[]
 for lab in [-1,*np.unique(labs)]:
  tr=labs!=lab; te=labs==lab
  beta=np.linalg.solve(F[tr].T@F[tr]+ridge*np.eye(F.shape[1]), F[tr].T@y[tr]) if ridge else np.linalg.lstsq(F[tr],y[tr],rcond=None)[0]
  co.append(np.linalg.lstsq(A.T,beta.T,rcond=None)[0].T)
  pred[te]=F[te]@co[-1]@A
 e=pred-y; return np.sqrt(np.mean(e[:,:-1]**2)),np.sqrt(np.mean(e[:,-1]**2)),max(np.sqrt(np.mean(e[labs==l,:-1]**2)) for l in np.unique(labs)),max(np.sqrt(np.mean(e[labs==l,-1]**2)) for l in np.unique(labs)),np.sqrt(np.mean((F@co[0]@A-y)**2)),co[0]
xdd=m._features(d,d)[:nd]; xo=m._features(tr,d)[n:]; xall=m._features(tr,{k:np.concatenate([te[k],d[k]]) for k in ('lab_id','stratum','coupon_features_raw','ref_coupons')})[-nd:]; ref=np.vstack([tr['ref_coupons'],te['ref_coupons']]); labref=np.r_[tr['lab_id'],te['lab_id']]; xfixed=d['coupon_features_raw'].copy()
for l in np.unique(labs):
 p=ref[labref==l]; xfixed[labs==l]=(xfixed[labs==l]-np.median(p.mean(0)))/np.median(p.std(0))
xfixed=np.where(np.isnan(xfixed),x[n:].mean(0),xfixed)
for tag,xx in [('train+diag',xo),('diag',xdd),('all',xall),('fixed',xfixed)]:
 rr=th[2]+xx@th[3:]; print(tag, 'vs train+diag drift rms',np.sqrt(np.mean((rr-dr)**2)))
 for name,F in [('dr',np.c_[F0,rr]),('dr+memory',np.c_[F0,rr,Z]),('dr+history',np.c_[F0,rr,past]),('features',np.c_[F0,xx]),('dr+sq',np.c_[F0,rr,rr**2]),('feat+memory',np.c_[F0,xx,Z])]:
  *met,coef=cv(F); print(name, np.round(met,5), ('coef '+str(np.round(coef,4))) if name=='dr' else '')
print('test loads & drift ranges'); pt,ht=m._loads(te,b); print(np.quantile(ht,[0,.5,1]),np.quantile(h,[0,.5,1]),np.quantile(th[2]+x[n:]@th[3:],[0,.5,1]),np.quantile(dr,[0,.5,1])); print('diag missing',np.where(np.isnan(d['coupon_features_raw'])))
