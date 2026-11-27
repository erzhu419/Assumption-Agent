import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 from m99_rank2 import *
from scipy.optimize import least_squares
lab=g['lab_id']; st=g['stratum']; missing=np.isnan(g['coupon_features_raw']); print('missing locations',np.argwhere(missing).tolist()); print('eval ranges by lab in test',[(l,int(sum(te['lab_id']==l))) for l in np.unique(te['lab_id'])]);
for name,blocks in [('tr+g',[tr,g]),('tr+te+g',[tr,te,g]),('tr+te',[tr,te]),('g',[g])]:
 ref=np.vstack([v['ref_coupons'] for v in blocks]); labs=np.concatenate([v['lab_id'] for v in blocks]); strata=np.concatenate([v['stratum'] for v in blocks]); x=np.vstack([v['coupon_features_raw'] for v in blocks]); G0=g['coupon_features_raw'].copy();
 for l in np.unique(labs):
  p=ref[labs==l]; c=np.median(p.mean(0)); scale=np.median(p.std(0)); x[labs==l]=(x[labs==l]-c)/scale; G0[lab==l]=(G0[lab==l]-c)/scale
 _,i,k=np.unique(np.c_[labs,strata],axis=0,return_inverse=True,return_counts=True); w=1/k[i]; imp=np.nansum(x*w[:,None],0)/np.sum(np.isfinite(x)*w[:,None],0); X0=np.where(missing,imp,G0)
 P0,H0,D0=m._coupon_inputs(g,X0,bridge); r=D0@theta[2:]; z=r-theta[2]; Y=E+drift[:,None]*T; b0=basis(1.)[:,0]; amp=Y@b0/(b0@b0); I=np.eye(4)[lab]; c=~missing.any(1); print('\n',name,'impute',imp.round(4));
 for l in range(4):
  j=(lab==l); jj=j&c; A=np.c_[np.ones(len(z)),z]; d=np.linalg.lstsq(A[jj],amp[jj],rcond=None)[0]; print('lab',l,'affine on complete',d.round(5),'rows [stratum centered_drift eff_gain hold_diff error missing]',np.c_[st,z,(amp-theta[2])/np.where(abs(z)>.001,z,np.nan),hc[:,0]-hc[:,3],amp-A@d,~c][j].round(4).tolist())
 if name=='tr+g':
  A=np.c_[np.ones(len(z)),I*z[:,None]]; B=np.c_[A,(hc[:,0]-hc[:,3])*z**2]; C=np.c_[np.ones(len(z)),z,z**2,z**3,(hc[:,0]-hc[:,3])*z**2]; D=np.c_[I,I*z[:,None]]; F=np.c_[np.ones(len(z)),z];
  for label,M in [('gain',A),('gain+contrast',B),('contrast',C),('gain+offset',D),('global',F)]:
   print('model',label,'coef on complete',np.linalg.lstsq(M[c],amp[c],rcond=None)[0].round(5));
   for fit_complete in (False,True):
    j=c if fit_complete else np.ones(len(z),bool); err=np.empty(len(z)); es=np.empty(len(z));
    for k in range(len(z)):
     mask=j&(np.arange(len(z))!=k); err[k]=amp[k]-M[k]@np.linalg.lstsq(M[mask],amp[mask],rcond=None)[0]; mask=j&(st!=st[k]); es[k]=amp[k]-M[k]@np.linalg.lstsq(M[mask],amp[mask],rcond=None)[0]
    print('complete fit',fit_complete,'loo all/complete/missing, strata worst',*[round(np.sqrt(np.mean(err[q]**2)),5) for q in (slice(None),c,~c)],round(max(np.sqrt(np.mean(es[st==k]**2)) for k in np.unique(st)),5))
