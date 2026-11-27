exec(open('/tmp/probe_chamber.py').read().split('for tag,')[0])
import json
raw=d['coupon_features_raw']; ld=d['lab_id']; missing=np.isnan(raw).any(1); Y=np.c_[yc,rc]; age=np.cumsum(dt[::-1])[::-1]-dt
for a in json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')):
 print('apparatus',a['completed_campaign_short_name'],np.round(np.mean(a['facility_telemetry_trace'],axis=0),4));
for tag,rs,ls in [('joint',ref,lab),('diagnostic',d['ref_coupons'],ld),('train',tr['ref_coupons'],tr['lab_id']),('test',te['ref_coupons'],te['lab_id'])]:
 print('reference',tag,[(int(l),round(np.median(rs[ls==l].mean(0)),4),round(np.median(rs[ls==l].std(0)),4)) for l in np.unique(ls)])
 X=raw.copy()
 for l in np.unique(ld):
  R=rs[ls==l]; X[ld==l]=(X[ld==l]-np.median(R.mean(0)))/np.median(R.std(0))
 for imp in ('joint','diagnostic','zero'):
  _,iv,ct=np.unique(np.c_[ld,d['stratum']],axis=0,return_inverse=True,return_counts=True); wgt=1/ct[iv]; calc=np.nansum(X*wgt[:,None],axis=0)/(np.isfinite(X).T@wgt)
  Xc=np.where(np.isfinite(X),X,{'joint':np.mean(np.r_[x,xt],axis=0),'diagnostic':calc,'zero':0}[imp]); r=beta[2]+Xc@beta[3:]; F=np.c_[hc,r]
  def fit(ii):
   def proj(v):
    u=np.exp(v); T=np.r_[-np.expm1(-t[:,None]/u),[1/u]]; A=(T[None,:,:,None]*F[:,None,None,:]).reshape(len(Y),Y.shape[1],-1); B=A[ii].reshape(-1,4); c=np.linalg.lstsq(B,Y[ii].ravel(),rcond=None)[0]; return A@c-Y,c
   sol=least_squares(lambda v:proj(v)[0][ii].ravel(),np.log(tau),bounds=(np.log([.4,2.]),np.log([1.,9.])),max_nfev=45); e,c=proj(sol.x); return e,c,np.exp(sol.x)
  err,c,u=fit(~missing); oo=np.zeros_like(Y)
  for l in np.unique(ld):
   e,_,_=fit((ld!=l)&~missing); oo[ld==l]=e[ld==l]
  m=lambda e:np.round(np.sqrt(np.mean(e**2,axis=0))[[-2,-1]],5)
  print(tag,imp,'tau',np.round(u,4),'coef',np.round(c,4),'complete-fit',m(err[~missing]),'complete-OOF',m(oo[~missing]),'missing-OOF',m(oo[missing]),'worst lab',np.max([m(oo[(ld==l)&~missing]) for l in np.unique(ld)],axis=0))
