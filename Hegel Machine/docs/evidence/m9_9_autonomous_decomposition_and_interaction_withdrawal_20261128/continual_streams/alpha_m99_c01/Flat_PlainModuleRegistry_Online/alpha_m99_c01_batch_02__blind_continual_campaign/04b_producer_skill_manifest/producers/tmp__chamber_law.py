import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 from drift_probe import *
_,xt=old._features(tr,te)
def aligned(part):
 xx=part['coupon_features_raw'].copy()
 for l in np.unique(part['lab_id']):
  m=part['lab_id']==l; rr=part['ref_coupons'][m]; xx[m]=(xx[m]-np.median(rr.mean(0)))/np.median(rr.std(0))
 return xx

def center(s,xx):
 _,j,n=np.unique(np.c_[s['lab_id'],s['stratum']],axis=0,return_inverse=True,return_counts=True); w=1/n[j]; return np.nansum(xx*w[:,None],axis=0)/(np.isfinite(xx).T@w)
zz=aligned(diag); zt=aligned(te); centers={'diag_session':center(diag,zz),'deploy_session':center(te,zt),'deploy_diag_scale':center(te,transforms([diag],[te])) if False else None,'joint_bench_deploy':center(te,np.where(np.isfinite(zt),xt,np.nan))}
rt=te['coupon_features_raw'].copy()
for l in np.unique(te['lab_id']):
 rr=diag['ref_coupons'][diag['lab_id']==l]; m=te['lab_id']==l; rt[m]=(rt[m]-np.median(rr.mean(0)))/np.median(rr.std(0))
centers['deploy_diag_scale']=center(te,rt)
for name,mu in centers.items():
 v=b[2]+np.where(np.isfinite(zz),zz,mu)@b[3:]; M=v[:,None,None]*F.T; print(name,'center',mu.round(4));
 for model in ['modal','conserved derivative','rounded']:
  yh=np.zeros_like(Y)
  for l in np.unique(labs):
   m=labs!=l; w=np.r_[np.ones(len(t)),2.]; W=M[m]*w[None,:,None]; yy=Y[m]*w
   if model=='modal': q=np.linalg.lstsq(W.reshape(-1,3),yy.ravel(),None)[0]
   elif model=='conserved derivative':
    A=W[:,:,:2]-W[:,:,2,None]/tau; q=np.linalg.lstsq(A.reshape(-1,2),(yy-W[:,:,2]).ravel(),None)[0]; q=np.r_[q,1-np.sum(q/tau)]
   else: q=np.array([.48,2.6,-.26])
   yh[~m]=M[~m]@q
  err=yh-Y; print(model,q.round(5),max(np.sqrt(np.mean(err[labs==l,:-1]**2)) for l in set(labs)),max(np.sqrt(np.mean(err[labs==l,-1]**2)) for l in set(labs)),np.sqrt(np.mean(err[:,:-1]**2)),np.sqrt(np.mean(err[:,-1]**2)), np.sqrt(np.mean(err[:,-2]**2)))
print('held-out predictor impact of deployment vs joint scaling:',np.std((np.where(np.isfinite(zt),zt,centers['deploy_session'])-xt)@b[3:]))
