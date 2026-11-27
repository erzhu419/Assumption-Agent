exec(open('/tmp/chamber_probe.py').read().split('for block,features')[0])
P,H,X=inp(diag,xd); T,Y=obs(diag); Q=np.c_[T,Y]; r=-np.expm1(-t[:,None]/taus); K=np.c_[np.vstack([r,1/taus]),np.r_[t,1.]]; complete=np.isfinite(diag['coupon_features_raw']).all(1); age=np.cumsum(dt[::-1])[::-1]-dt; W=-np.expm1(-dt[:,None]/taus)*np.exp(-age[:,None]/taus); print('nominal history weights',W.T.round(5)); print('complete',complete.sum());
for tail in [True, False]:
 k=K if tail else K[:,:2]; coeff=Q@np.linalg.pinv(k.T); err=Q-coeff@k.T; print('temporal linear?',tail,'curve/rate rmse',np.sqrt(np.mean(err**2,0)).round(5),'sv residual',np.linalg.svd(err,compute_uv=False).round(5));
 print('coeff projected ranges',coeff.min(0), coeff.max(0));
 # Estimate history contrasts independently of any coupon feature transform by projecting out each candidate drift kernel.
 for mix in [(0.5,0.5,0),(0,1,0),(1,0,0),(0,0,1),(.4,.6,0),(.5,.5,.15)]:
  v=np.array(mix)*np.r_[taus,1]; g=k@v[:k.shape[1]]; G=np.eye(len(g))-np.outer(g,g)/(g@g); B=np.c_[P,H]; mech=[np.outer(f,G@K[:,i]) for i in range(2) for f in B.T]; A=np.stack(mech,axis=2).reshape(-1,14); tar=(Q@G).ravel(); fit=np.linalg.lstsq(A,tar,rcond=1e-8)[0]; print('kernel',mix,'rmse',np.sqrt(np.mean((A@fit-tar)**2)).round(5), 'coeffs',fit.reshape(2,7).round(4))
 print('row residual norm',np.linalg.norm(err,axis=1).round(4))
 if tail:
  # Check if a temporal direction outside the two intrinsic exponentials and linear drift is actually measurable.
  for label,M in [('diag',Q),('complete',Q[complete])]:
   print(label,'response singular values',np.linalg.svd(M,compute_uv=False).round(5)); print('largest temporal residual modes',np.linalg.svd((M-M@np.linalg.pinv(K.T)@K.T),full_matrices=False)[2][:2].round(3))
