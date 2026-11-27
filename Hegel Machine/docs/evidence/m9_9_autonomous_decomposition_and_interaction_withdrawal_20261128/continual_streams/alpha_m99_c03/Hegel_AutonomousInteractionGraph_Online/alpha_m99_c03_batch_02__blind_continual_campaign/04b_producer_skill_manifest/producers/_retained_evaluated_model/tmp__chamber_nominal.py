exec(open('/tmp/chamber_probe.py').read().split('for block,features')[0])
P,H,_=inp(diag,xd); T,Y=obs(diag); Q=np.c_[T,Y]; X=q._features(diag,diag)[:len(P)]; D=theta[2]+X@theta[3:]; labs=diag['lab_id']; complete=np.isfinite(diag['coupon_features_raw']).all(1); age=np.cumsum(dt[::-1])[::-1]-dt
K=np.r_[-np.expm1(-t[:,None]/taus),[1/taus]]; W=-np.expm1(-dt[:,None]/taus)*np.exp(-age[:,None]/taus); G=(H[:,None]-P@W)*theta[:2]; B=G@K.T; k=np.r_[-np.expm1(-t),1.]; E=Q-B-D[:,None]*k; print('nominal residual strain/rate RMS',np.sqrt(np.mean(E[:,:-1]**2)),np.sqrt(np.mean(E[:,-1]**2))); print('by lab',[(int(l),np.sqrt(np.mean(E[labs==l,:-1]**2)),np.sqrt(np.mean(E[labs==l,-1]**2))) for l in np.unique(labs)]); print('complete/missing',[(np.sqrt(np.mean(E[m,:-1]**2)),np.sqrt(np.mean(E[m,-1]**2))) for m in [complete,~complete]]); print('row drift error',np.round(E@k/(k@k),5));
J=np.eye(len(k))-np.outer(k,k)/(k@k); print('orthogonal error',np.sqrt(np.mean((E@J)**2)),np.linalg.svd(E@J,compute_uv=False)[:4]);
for mode in ['nominal','history_scale','history_tau','history_tau_scale','free_history']:
 def err(p,ptrain,full=False):
  drift=p[-1]*(-np.expm1(-t/p[-1])); k=np.r_[drift,1.]; J=np.eye(len(k))-np.outer(k,k)/(k@k); Ht=B; A=None
  if mode in ['history_tau','history_tau_scale']:
   mt=np.array(p[:2]); wt=-np.expm1(-dt[:,None]/mt)*np.exp(-age[:,None]/mt); Z=(P@wt)*theta[:2]; Ht=(H[:,None]*theta[:2]-Z)@K.T
  else: Z=(P@W)*theta[:2]
  if mode in ['history_scale','history_tau_scale']: A=np.stack([np.outer(Z[:,i],K[:,i]) for i in range(2)],-1)
  if mode=='free_history': A=np.stack([np.outer(z,h) for h in K.T for z in P.T],-1)
  if A is not None:
   Aj=np.einsum('ntl,tu->nul',A,J); c=np.linalg.lstsq(Aj[ptrain].reshape(-1,A.shape[-1]),((Q-Ht)@J)[ptrain].ravel(),rcond=None)[0]; Ht=Ht+A@c
  r=(Q-Ht)@J
  return (r,(Q-Ht)@k/(k@k)-D,c if A is not None else []) if full else r[ptrain].ravel()
 use_tau='history_tau' in mode; p0=np.r_[taus,1.] if use_tau else [1.]
 def fit(m): return least_squares(lambda p:err(p,m),p0,bounds=([.2,.4,.5],[2.,10.,2.]) if use_tau else (.5,2.)).x
 p=fit(np.ones(len(Q),bool)); r,de,c=err(p,None,True); cv=[]
 for l in np.unique(labs):
  m=labs==l; rr,_,_=err(fit(~m),None,True); cv.append(np.sqrt(np.mean(rr[m]**2)))
 print(mode,'p',p.round(5),'c',np.round(c,5),'orth rmse',np.sqrt(np.mean(r*r)),'worst lab orth',max(cv),'drift complete/missing RMS',[np.sqrt(np.mean(de[m]**2)) for m in [complete,~complete]])
