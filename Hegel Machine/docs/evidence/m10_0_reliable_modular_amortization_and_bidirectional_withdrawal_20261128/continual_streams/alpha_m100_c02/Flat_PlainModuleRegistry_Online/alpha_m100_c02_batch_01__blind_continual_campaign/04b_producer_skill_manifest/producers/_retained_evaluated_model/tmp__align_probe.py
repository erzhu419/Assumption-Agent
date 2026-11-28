from chamber_probe import *
for name,v in [('joint',X[:len(H)]),('separate',m._features(tr,tr)[:len(H)])]:
 yy=tr['strain_curves_obs']-tr['strain_0'][:,None]; rr=tr['instant_rate_obs']; u,q=m._fit(P,H,v,dt,t,yy,rr); a,b,*_=m._design(n.log(u),P,H,v,dt,t)
 print('train',name,u.round(5),q.round(5),'curve abs median',n.median(abs(a@q-yy)),'rate abs median',n.median(abs(b@q-rr)))
raw=g['coupon_features_raw'].copy(); lab=g['lab_id']; strata=g['stratum']; refs=g['ref_coupons'];
for l in n.unique(lab):
 ix=lab==l; raw[ix]=(raw[ix]-n.median(refs[ix].mean(0)))/n.median(refs[ix].std(0))
for axis in ('row','lab','stratum','cell'):
 keys={'row':n.arange(len(h)), 'lab':lab, 'stratum':strata,'cell':n.c_[lab,strata]}[axis]; _,i,cnt=n.unique(keys,axis=0,return_inverse=True,return_counts=True); w=1/cnt[i]; print('diagnostic fill',axis,n.nansum(raw*w[:,None],0)/(~n.isnan(raw)).T.dot(w))
