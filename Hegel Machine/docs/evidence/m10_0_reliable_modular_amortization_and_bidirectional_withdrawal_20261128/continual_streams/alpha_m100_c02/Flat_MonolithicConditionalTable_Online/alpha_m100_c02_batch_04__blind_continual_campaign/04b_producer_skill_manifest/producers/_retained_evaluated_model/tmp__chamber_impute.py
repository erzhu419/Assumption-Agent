exec(open('/tmp/chamber_probe.py').read().split('for tag,s,h')[0])
Y=v['strain_curves_obs']-v['strain_0'][:,None]; R=v['instant_rate_obs']; Q=np.c_[Y,R]; n=len(R)
B=np.vstack([np.c_[-np.expm1(-t[:,None]/tau),t],np.r_[1/tau,1]])
base=(hv[:,None]*theta[:2])@B[:,:2].T
F=v['coupon_features_raw'].astype(float).copy(); lab=v['lab_id']; st=v['stratum']; refs=v['ref_coupons']; complete=np.isfinite(F).all(1)
for j in np.unique(lab):
 k=lab==j; F[k]=(F[k]-np.median(refs[k].mean(0)))/np.median(refs[k].std(0))
_,cells,counts=np.unique(np.c_[lab,st],axis=0,return_inverse=True,return_counts=True); w=1/counts[cells].reshape(-1,1)
fill=lambda a: np.where(np.isfinite(F),F,a)
variants={'balanced':fill(np.nansum(F*w,axis=0)/np.sum(np.isfinite(F)*w,axis=0)),'row_mean':fill(np.nanmean(F,axis=0)),'zero':fill(0),'lab_mean':fill(np.array([np.nanmean(F[lab==j],axis=0) for j in lab])),'stratum_mean':fill(np.array([np.nanmean(F[st==j],axis=0) for j in st]))}
folds=[v[k]==c for k in ('lab_id','stratum') for c in np.unique(v[k])]
for mode,h in [('balanced',variants['balanced']),*[(k,f) for k,f in variants.items() if k!='balanced']]:
 D=theta[2]+h@theta[3:]; print(mode)
 for use in ['all','complete']:
  trainmask=complete if use=='complete' else np.ones(n,bool)
  def fit(k):
   co=np.linalg.lstsq(D[k,None],(Q-base)[k],rcond=None)[0]@np.linalg.pinv(B.T)
   return co[0]
  def report(e):return np.round([np.sqrt(np.mean(e[...,:-1]**2)),np.sqrt(np.mean(e[...,-1]**2)),np.max(abs(e[...,:-1])),np.max(abs(e[...,-1]))],6)
  co=fit(trainmask); e=base+D[:,None]*(co@B.T)-Q
  worst=np.max([report((base+D[:,None]*(fit((~k)&trainmask)@B.T)-Q)[k]) for k in folds],axis=0)
  print(use,co.round(7),'worst',worst,'complete vs missing',report(e[complete]),report(e[~complete]))
  if mode=='balanced' and use=='complete': print('missing inferred drift vs imputed',np.c_[np.isnan(F[~complete]).argmax(1),D[~complete], ((Q-base)[~complete]@(co@B.T))/((co@B.T)@(co@B.T))].round(5))
print('test pooled vs local drift quantiles',*[np.quantile(q@theta[2:],[0,.25,.5,.75,1]).round(4) for q in (x[len(ht):],m._features(te,te)[:len(te['lab_id'])])])
