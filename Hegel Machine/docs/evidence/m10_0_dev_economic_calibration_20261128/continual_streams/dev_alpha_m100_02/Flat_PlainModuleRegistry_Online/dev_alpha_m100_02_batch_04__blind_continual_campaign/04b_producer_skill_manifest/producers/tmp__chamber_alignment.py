import numpy as np
from scipy.optimize import least_squares
z=np.load('/tmp/chamber_explore.npz'); c=np.load('/tmp/chamber_params.npz'); y,B,a,f,h,tau,th=c['y'],c['B'],c['a'],c['f'],c['h'],c['tau'],c['th']; reset=c['reset'].copy();reset[:,2]=0
ps=[{k:z[n+'_'+k] for k in ('lab_id','stratum','ref_coupons','coupon_features_raw')} for n in ('tr','te','q')];q=ps[-1]; labs=q['lab_id']; raw=q['coupon_features_raw']; print('missing diagnostic rows:',np.where(np.isnan(raw).any(1))[0]); print('lab missing a_slow f raw:',np.c_[labs,np.isnan(raw).sum(1),(a-reset)[:,1],f,raw].round(3))
for mode,parts in [('pooled',ps[:2]),('diagnostic',ps[-1:]),('all',ps),('train',ps[:1]),('test',ps[1:2])]:
 lab=np.concatenate([p['lab_id'] for p in parts]); ref=np.concatenate([p['ref_coupons'] for p in parts]); rr=np.concatenate([p['coupon_features_raw'] for p in parts]); strata=np.concatenate([p['stratum'] for p in parts]); x=raw.copy();print(mode,end=' ')
 for l in np.unique(lab):
  refs=ref[lab==l]; m,s=np.median(refs.mean(0)),np.median(refs.std(0));print(np.round([m,s],3),end=' ');x[labs==l]=(x[labs==l]-m)/s;rr[lab==l]=(rr[lab==l]-m)/s
 _,j,n=np.unique(np.c_[lab,strata],axis=0,return_inverse=True,return_counts=True); w=1/n[j]; mi=np.nansum(w[:,None]*rr,0)/(w@np.isfinite(rr));x=np.where(np.isnan(x),mi,x); f=th[2]+x@th[3:];Y=y-reset@B.T
 for name,F in [('force',f[:,None]),('free',np.c_[np.ones(len(x)),x])]:
  es=[]
  for l in np.r_[np.unique(labs),-1]:
   sel=labs!=l
   v=least_squares(lambda u:(F[sel]@u.reshape(-1,3)@B.T-Y[sel]).ravel(),np.linalg.lstsq(F[sel],(a-reset)[sel],rcond=None)[0].ravel(),loss='soft_l1',f_scale=.008).x.reshape(-1,3)
   if l==-1: print(name,'coef',v.round(4).tolist(),'worst',np.max(es,0).round(4));break
   e=F[~sel]@v@B.T-Y[~sel]; es.append([np.sqrt(np.mean(e[:,:-1]**2)),np.sqrt(np.mean(e[:,-1]**2))])
