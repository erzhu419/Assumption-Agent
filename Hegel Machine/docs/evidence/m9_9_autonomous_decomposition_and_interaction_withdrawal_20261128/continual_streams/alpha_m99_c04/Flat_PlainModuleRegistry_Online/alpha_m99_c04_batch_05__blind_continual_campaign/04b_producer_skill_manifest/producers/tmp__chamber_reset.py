exec(open('/tmp/chamber_probe.py').read().split('for name,xc')[0])
from itertools import product
Y=np.c_[c['instant_rate_obs'],c['strain_curves_obs']-c['strain_0'][:,None]]
S=c['past_channel_raw'].sum(-1); h=c['hold_channel_raw'].sum(-1); labs=c['lab_id']; missing=np.isnan(c['coupon_features_raw']).any(axis=1)
A=lambda ts: np.r_[1/ts,-np.expm1(-t/ts)]
L=np.r_[1,t]; base=h[:,None]*(theta[:2]@np.array([A(s) for s in tau]))
E=Y-base
U,sv,V=np.linalg.svd(E,full_matrices=False)
print('full reset residual singular values',sv[:6]); print('leading residual shape',np.round(V[0]/V[0,0],4)); print('linear shape',L)
# Regress each temporal sample on hold and a free coupon drift amplitude,
# with temporal orthogonality to the intrinsic branch basis for identification.
for inds in [np.arange(len(h)), np.where(~missing)[0]]:
 eh=E[inds]; hh=h[inds]; perp=eh-hh[:,None]*(hh@eh)/(hh@hh); u,s,v=np.linalg.svd(perp,full_matrices=False)
 print('reset projected SVD',s[:6]); print('drift shape',np.round(v[0]/v[0,0],4))
 print('hold component orthogonal to drift',np.round(hh@eh/(hh@hh)-(hh@eh/(hh@hh)@v[0])*v[0],5))
def cat(groups):
 return {k:np.concatenate([g[k] for g in groups]) for k in ('lab_id','stratum','ref_coupons','coupon_features_raw')}
contexts={k:_features(g,c)[1] for k,g in [('train',a),('test',b),('diag',c),('both',cat([a,b]))]}
allidx=np.arange(len(h))
for name,xc in contexts.items():
 drift=theta[2]+xc@theta[3:]; Vd=np.c_[np.ones(len(h)),drift]; modes={'linear':L,'unit_sat':A(1.),'slow_sat':tau[1]*A(tau[1]),'mixed':None}
 for mode,basis in modes.items():
  def fit(idx):
   if basis is None:
    f=np.c_[h,Vd]; co=np.linalg.lstsq(f[idx],E[idx],rcond=None)[0]; return f@co,co
   f=Vd[:,:,None]*basis; f=np.concatenate([f,(h[:,None]*A(tau[0]))[:,None,:],(h[:,None]*A(tau[1]))[:,None,:]],axis=1); D=f.transpose(0,2,1); co=np.linalg.lstsq(D[idx].reshape(-1,4),E[idx].ravel(),rcond=None)[0]; return D@co,co
  pred,co=fit(allidx); er=pred-E; worst=[]
  for lab in np.unique(labs):
   j=labs==lab; e=fit(allidx[~j])[0][j]-E[j]; worst.append([np.sqrt(np.mean(e[:,1:]**2)),np.sqrt(np.mean(e[:,0]**2))])
  print(name,mode,'rmse complete/missing',*[np.round(np.sqrt(np.mean(er[j]**2)),5) for j in (~missing,missing)],'worst curve/rate',np.round(np.max(worst,axis=0),5),'coef',np.round(co,4))
print('feature reference shapes',a['ref_coupons'].shape,b['ref_coupons'].shape,c['ref_coupons'].shape)
