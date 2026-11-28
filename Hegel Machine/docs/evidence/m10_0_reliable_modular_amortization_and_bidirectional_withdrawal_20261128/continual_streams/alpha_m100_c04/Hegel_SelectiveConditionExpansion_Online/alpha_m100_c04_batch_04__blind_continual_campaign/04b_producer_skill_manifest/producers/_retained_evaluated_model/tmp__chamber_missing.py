exec(open('/tmp/chamber_probe.py').read().split('for tt in (ta,')[0])
import downstream_solver as s
f=F(ta); k=f@s._CHAMBER_COUPLING; rr=Y-h[:,None]*(f[:,:2]@th[:2]); observed=rr@k/(k@k)
raw=d['coupon_features_raw']; lab=d['lab_id']; st=d['stratum']; panel=d['ref_coupons']; missing=~np.isfinite(raw); x=np.empty_like(raw)
for l in np.unique(lab):
 m=lab==l; x[m]=(raw[m]-np.median(panel[m].mean(0)))/np.median(panel[m].std(0))
_,g,c=np.unique(np.column_stack((lab,st)),axis=0,return_inverse=True,return_counts=True); w=1/c[g]; full=~missing.any(1)
for nm,mu in [('observed inverse-cell-count',np.nansum(x*w[:,None],0)/(w@~missing)),('unweighted',np.nanmean(x,0)),('balanced featurewise cell means',np.nanmean(np.array([np.nansum(x[g==i],axis=0)/np.maximum(1,np.sum(~missing[g==i],axis=0)) for i in range(len(c))]),axis=0)),('standardized zero',np.zeros(x.shape[1]))]:
 inp=th[2]+np.where(missing,mu,x)@th[3:]; print(nm,np.round(mu,5),'missing drift error',np.round((observed-inp)[~full],5))
for i in np.flatnonzero(~full):
 cols=np.flatnonzero(missing[i]); print('missing row',i,'lab/stratum',lab[i],st[i],'cols',cols,'inferred missing contribution',observed[i]-th[2]-np.nansum(x[i]*th[3:]))
