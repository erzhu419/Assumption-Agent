exec(open('/tmp/chamber_law.py').read().split('for label,F in')[0])
print('t/dt',t,dt,'keys',list(d),list(c))
print('missing and labs',np.c_[c['lab_id'],np.isnan(c['coupon_features_raw']).astype(int)])
raw=c['coupon_features_raw']; labs=c['lab_id']; ref=c['ref_coupons'];
for l in np.unique(labs):
 p=ref[labs==l]; print('panel',l,'col location',p.mean(0).round(3),'scale',p.std(0).round(3))
# Profile a common spectrum using the shared hold-and-drift features. Compare
# aligned panels, targeted history corrections, and nonlinear drift terms.
dv=xc@theta[3:]; ht=hc-hc.mean(); z=hc[:,None]-tau*m._design(tau,xc,pc,hc,dt,t)[1][:,:2];
F0=np.c_[np.ones(n),hc,dv]; wt=.5

def test(F,D,freedom,ix,loss='soft_l1'):
 def mat(tt):
  B=np.vstack((np.c_[-np.expm1(-t[:,None]/tt),t],wt*np.r_[1/tt,1]))
  return np.concatenate([f[:,None,:]*B[None,:,j,None] for j,f in enumerate((*F,D))],axis=2)
 p0=np.log(tau); a0=mat(tau); beta=np.linalg.lstsq(a0[ix].reshape(-1,a0.shape[-1]),Y[ix].ravel(),rcond=None)[0]; nz=len(beta)
 def calc(p):
  pp=p[nz:]; tt=np.exp(p0+(pp if freedom==2 else [pp[0],0])) if freedom else tau
  return mat(tt),tt
 def fun(p):
  A,_=calc(p);return (A[ix]@p[:nz]-Y[ix]).ravel()
 lo=np.r_[np.full(nz,-np.inf),np.full(freedom,-.3)]; hi=np.r_[np.full(nz,np.inf),np.full(freedom,.3)]
 q=least_squares(fun,np.r_[beta,np.zeros(freedom)],bounds=(lo,hi),loss=loss,f_scale=.0025,x_scale='jac',max_nfev=80).x
 A,tt=calc(q); return A@q[:nz]-Y,tt,q[:nz]
D=np.c_[np.ones(n),dv]; alt=[('baseline',(F0,F0),D),('history matched',(np.c_[F0,z[:,0]],np.c_[F0,z[:,1]]),D),('history cross',(np.c_[F0,z],np.c_[F0,z]),D),('nonlinear drift',(np.c_[F0,dv**2],np.c_[F0,dv**2]),np.c_[D,dv**2]),('no drift',(F0,F0),D[:,:0]),('drift through origin',(F0,F0),(theta[2]+dv)[:,None])]; indices=np.arange(n)
for name,F,DD in alt:
 for free in range(3):
  e,tt,beta=test(F,DD,free,indices); cv=np.array([test(F,DD,free,indices[indices!=i])[0][i] for i in indices]); lo=np.zeros_like(e)
  for l in np.unique(labs):
   mask=labs==l; lo[mask]=test(F,DD,free,indices[~mask])[0][mask]
  print(name,free,'loo',report(cv),'lolo',report(lo),'taus',tt.round(5),'coeff',beta.round(5))
  if name=='baseline' and free==2: print('per coupon LOO curve/rate',np.c_[np.sqrt(np.mean(cv[:,:-1]**2,axis=1)),2*cv[:,-1]].round(5))
