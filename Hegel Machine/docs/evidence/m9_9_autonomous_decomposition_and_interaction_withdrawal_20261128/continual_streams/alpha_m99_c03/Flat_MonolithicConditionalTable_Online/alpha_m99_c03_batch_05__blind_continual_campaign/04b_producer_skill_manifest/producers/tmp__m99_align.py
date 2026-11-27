from m99_probe import *
from scipy.optimize import least_squares
from collections import defaultdict
B=hold[:,None]*th[:2]; branch=B@bas.T; rr=r-B@(1/tau); G=2*np.expm1(.5*np.log1p(t)); u=((y-branch)@G+rr)/(G@G+1)
# Test which reference population explains diagnostic drift, keeping the intrinsic fit fixed.
def align(target, blocks):
 raw=target['coupon_features_raw']; out=raw.copy(); ref=np.vstack([z['ref_coupons'] for z in blocks]); lab=np.concatenate([z['lab_id'] for z in blocks]); strata=np.concatenate([z['stratum'] for z in blocks]); feats=np.vstack([z['coupon_features_raw'] for z in blocks]); print('panel',len(ref),end=' ')
 for k in np.unique(target['lab_id']):
  pan=ref[lab==k]; mu=np.median(pan.mean(0)); sd=np.median(pan.std(0)); i=target['lab_id']==k; out[i]=(raw[i]-mu)/sd; feats[lab==k]=(feats[lab==k]-mu)/sd; print(k,np.round([mu,sd],3),end='; ')
 _,cell,count=np.unique(np.c_[lab,strata],axis=0,return_inverse=True,return_counts=True); w=1/count[cell,None]; fill=np.nansum(w*feats,axis=0)/np.sum(w*np.isfinite(feats),axis=0); print('fill',np.round(fill,3)); return np.where(np.isfinite(out),out,fill)
def report(name, pred):
 metrics(name,branch+pred[:,None]*G,B@(1/tau)+pred)
 print(' drift error by lab',[(int(k),np.round([np.mean((pred-u)[i]),np.std((pred-u)[i])],5)) for k in np.unique(d['lab_id']) for i in [d['lab_id']==k]])
for name,blocks in [('diag',[d]),('train',[tr]),('test',[te]),('train+diag',[tr,d]),('train+test',[tr,te]),('all',[tr,te,d]),('test+diag',[te,d])]:
 v=align(d,blocks); report(name,th[2]+v@th[3:]);
print('residuals and candidate factors: lab stratum drift delta missing x hold memory'); print(np.round(np.c_[d['lab_id'],d['stratum'],drift,u-drift,np.isnan(d['coupon_features_raw']).sum(1),xd,hold,mem],4))
# Check whether previous alternate implementations contain a different feature or chamber rule.
