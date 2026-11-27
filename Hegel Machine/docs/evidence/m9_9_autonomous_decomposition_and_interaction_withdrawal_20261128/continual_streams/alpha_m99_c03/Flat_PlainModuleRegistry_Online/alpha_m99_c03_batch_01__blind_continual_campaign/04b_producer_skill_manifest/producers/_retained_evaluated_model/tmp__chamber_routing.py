exec(open('/tmp/chamber_memory.py').read().split("print('zero-drift")[0])
from scipy.optimize import least_squares
print('zero-drift free taus',q,'strain/rate rms',np.sqrt(np.mean(prof(q).reshape(N,-1)**2,axis=0))[[0,-1]])
print('missing features train/test/diagnostics',*[np.isnan(g['coupon_features_raw']).sum() for g in [tr,te,dg]])
print('zero-drift slopes against benchtop drift',np.linalg.lstsq(d@th[2:,None],a-h[:,None]*th[:2],rcond=None)[0]); print('taus train',taus)
after=np.cumsum(dt[::-1])[::-1]-dt
z=p@(-np.expm1(-dt[:,None]/taus)*np.exp(-after[:,None]/taus))
r=d@th[2:]; T=np.r_[t,1.]
for name in ('redirect','redirect_fit_taus','redirect_drift','redirect_fit_taus_drift','redirect_hold','redirect_features','redirect_features_drift','redirect_memory','redirect_nonlinear','redirect_coupled_decay'):
 def design(q):
  B=basis(q); base=(h[:,None]*th[:2])@B.T
  F=[B[None,:,j,None]*r[:,None,None] for j in range(2)]
  if 'features' in name: F=[B[None,:,j,None]*d[:,None,:] for j in range(2)]
  if name=='redirect_hold': F=[B[None,:,j,None]*np.c_[r,h,np.ones(N)][:,None,:] for j in range(2)]
  if name=='redirect_memory': F=[B[None,:,j,None]*np.c_[r,z][:,None,:] for j in range(2)]
  if name=='redirect_nonlinear': F=[B[None,:,j,None]*np.c_[r,r*r,r*h,r*z[:,0],r*z[:,1]][:,None,:] for j in range(2)]
  if name=='redirect_coupled_decay': F=[r[:,None,None]*np.c_[B[:,0],B[:,1],T][None,:,:]]
  if name.endswith('_drift'): F.append(r[:,None,None]*T[None,:,None])
  return base,np.concatenate(F,axis=-1)
 def learn(j):
  def profile(q,ret=False):
   base,F=design(q); X=F[j].reshape(len(j)*y.shape[1],-1); v=np.linalg.lstsq(X,(y-base)[j].ravel(),rcond=None)[0]; pred=base+F@v
   return (v,pred) if ret else (pred-y)[j].ravel()
  q=least_squares(profile,taus,bounds=(.05,25)).x if 'fit_taus' in name else taus
  v,pred=profile(q,True); return q,v,pred-y
 def metrics(E): return np.sqrt([np.mean(E[:,:-1]**2),np.mean(E[:,-1]**2)])
 folds=np.array([metrics(learn(ix[lab!=l])[2][lab==l]) for l in np.unique(lab)])
 q0,v,E=learn(ix)
 print(name,'rms',metrics(E).round(5),'worst LOLO',folds.max(0).round(5),'q',q0.round(5),'coef',v.round(5))
print('free drift profile at intrinsic taus:')
B=np.c_[basis(taus),T]; coeff=np.linalg.lstsq(B,y.T,rcond=None)[0].T; print(np.c_[r,coeff[:,:2]-h[:,None]*th[:2],coeff[:,2]].round(4))
