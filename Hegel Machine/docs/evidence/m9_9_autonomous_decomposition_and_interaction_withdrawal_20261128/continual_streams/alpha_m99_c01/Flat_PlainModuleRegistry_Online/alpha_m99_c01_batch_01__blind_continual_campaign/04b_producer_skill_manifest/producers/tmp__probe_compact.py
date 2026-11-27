exec(open('/tmp/probe_chamber.py').read().split('for tag,')[0])
raw=d['coupon_features_raw']; ld=d['lab_id']; X=raw.copy(); missing=~np.isfinite(X); Y=np.c_[yc,rc]; age=np.cumsum(dt[::-1])[::-1]-dt
for l in np.unique(ld):
 R=d['ref_coupons'][ld==l]; X[ld==l]=(X[ld==l]-np.median(R.mean(0)))/np.median(R.std(0))
_,iv,ct=np.unique(np.c_[ld,d['stratum']],axis=0,return_inverse=True,return_counts=True); ww=1/ct[iv]; center=np.nansum(X*ww[:,None],axis=0)/(np.isfinite(X).T@ww); print('diagnostic center',center,'missing',np.argwhere(missing).tolist()); print('times',dt,t); X=np.where(missing,center,X); r=beta[2]+X@beta[3:]; q=np.r_[t,1.]; z=pc@(np.exp(-age[:,None]/tau)*-np.expm1(-dt[:,None]/tau)); complete=~missing.any(1)

def fit(ii,kind):
 def project(v):
  u=tau*np.exp(v) if len(v) else tau
  T=np.r_[-np.expm1(-t[:,None]/u),[1/u]]
  F=np.stack([hc,r],axis=-1); A=(T[None,:,:,None]*F[:,None,None,:]).reshape(len(Y),Y.shape[1],4)
  if 'drift' in kind:A=np.concatenate([A,q[None,:,None]*F[:,None,:]],axis=-1)
  if 'history' in kind:A=np.concatenate([A,T[None,:,:]*z[:,None,:]],axis=-1)
  B=A[ii].reshape(-1,A.shape[-1]); c=np.linalg.lstsq(B,Y[ii].ravel(),rcond=None)[0]; return A@c-Y,c
 v=least_squares(lambda v:project(v)[0][ii].ravel(),[0.,0.],bounds=(-.4,.4),max_nfev=45).x if 'shift' in kind else []
 e,c=project(v); return e,c,tau*np.exp(v) if len(v) else tau

def metr(e): return np.r_[np.sqrt(np.mean(e[:,:-1]**2)),np.sqrt(np.mean(e[:,-2]**2)),np.sqrt(np.mean(e[:,-1]**2))]
for kind in ['fixed','shift','fixed drift','shift drift','shift history','fixed drift history']:
 for restrict in [True,False]:
  ii=complete if restrict else np.ones(len(Y),bool)
  e,c,u=fit(ii,kind); oo=np.zeros_like(Y)
  for l in np.unique(ld):
   E,_,_=fit((ld!=l)&ii,kind); oo[ld==l]=E[ld==l]
  print(kind,restrict,'u',np.round(u,6),'c',np.round(c,6),'fit',np.round(metr(e[ii]),5),'worst complete lab',np.round(np.max([metr(oo[(ld==l)&complete]) for l in np.unique(ld)],axis=0),5),'missing OOF',np.round(metr(oo[~complete]),5))
  if kind=='shift' and restrict:
   T=np.r_[-np.expm1(-t[:,None]/u),[1/u]]; signal=T@c[1::2]; pred_r=r-e@signal/(signal@signal); print('inferred minus calculated missing drift',np.round((pred_r-r)[~complete],5)); print('inferred imputation',np.round(center[np.nonzero(missing)[1]]+(pred_r-r)[np.nonzero(missing)[0]]/beta[3+np.nonzero(missing)[1]],5))
