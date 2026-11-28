exec(open('/tmp/chamber_basis.py').read().split('for label,y')[0])
np.set_printoptions(precision=7,suppress=True)
for key,a in p.items(): print(key, list(a) if isinstance(a,dict) else type(a))
for name,q in [('diag',dg),('train+test', {k:np.concatenate([tr[k],te[k]]) for k in ('lab_id','stratum','ref_coupons','coupon_features_raw')})]:
 raw=q['coupon_features_raw'].copy(); labs=q['lab_id']; cells=np.c_[labs,q['stratum']]
 for l in np.unique(labs):
  j=labs==l; rr=q['ref_coupons'][j];raw[j]=(raw[j]-np.median(rr.mean(0)))/np.median(rr.std(0))
 print(name,'means',np.nanmean(raw,0),'zero cells',np.nanmean([np.nanmean(raw[np.all(cells==u,axis=1)],0) for u in np.unique(cells,axis=0)],0));print('expected old fill',np.unique(np.where(np.isnan(raw),m['_features'](q,q)[:len(raw)],np.nan),axis=0)[:5]);print('stress range',np.quantile(m['_loads'](q,b)[1],[0,0.5,1]) if name=='diag' else np.quantile(m['_loads'](te,b)[1],[0,.5,1]));
print('taus exact',tau,'w',w)
