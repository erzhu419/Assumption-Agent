exec(open('/tmp/explore_chamber.py').read().split('x=diag_features([g]);')[0])
x=diag_features([g]); past,hold=m._loads(g,b); curve,rate=m._basis(taus,past,hold,x,dt,t); base=np.c_[curve@theta,rate@theta]; y=np.c_[g['strain_curves_obs']-g['strain_0'][:,None],g['instant_rate_obs']]; drift=theta[2]+x@theta[3:]; memrate=rate[:,:2]@theta[:2]; lab=g['lab_id']; missing=np.isnan(g['coupon_features_raw']).any(1)
e=-np.expm1(-t[:,None]/taus); tb=np.vstack([np.c_[e,t],np.r_[1/taus,1]]); cons=tb[:,:2]-tb[:,2:]/taus
for mask in [~missing,missing]:
 print('rate residual',int(mask.sum()),np.round((y-base)[mask,-1],5),'drift span',np.round(np.sort(drift[mask]),3))
for name,fit_complete,use_rate_drift,basis in [('conserved-all',False,False,cons),('conserved-complete',True,False,cons),('conserved-rate-proxy',False,True,cons),('free-complete',True,False,tb)]:
 errors=np.zeros_like(y)
 for held in [None,*np.unique(lab)]:
  fit=(lab!=held) if held is not None else np.ones(len(lab),bool)
  if fit_complete: fit &= ~missing
  df=(y[:,-1]-memrate) if use_rate_drift else drift
  B=base.copy(); B[:,:-1]+=(df-drift)[:,None]*t; B[:,-1]+=df-drift
  cols=slice(None,-1) if use_rate_drift else slice(None)
  A=(df[fit,None,None]*basis[cols]).reshape(-1,basis.shape[-1]); target=(y-B)[fit,cols].ravel(); q=least_squares(lambda q:A@q-target,np.linalg.lstsq(A,target,rcond=None)[0],loss='soft_l1',f_scale=.003).x
  if held is None: print(name,'q=',q,'rate shift/ unit drift=',basis[-1]@q)
  else: errors[lab==held]=(base+drift[:,None]*(basis@q)-y)[lab==held]
 print(' worst lab curve/rate',np.max([np.mean(abs(errors[lab==l]),axis=0) for l in np.unique(lab)],axis=0)[[0,-2,-1]].round(5),'complete curve/rate',abs(errors[~missing,:-1]).mean(),abs(errors[~missing,-1]).mean(),'missing curve/rate',abs(errors[missing,:-1]).mean(),abs(errors[missing,-1]).mean())
