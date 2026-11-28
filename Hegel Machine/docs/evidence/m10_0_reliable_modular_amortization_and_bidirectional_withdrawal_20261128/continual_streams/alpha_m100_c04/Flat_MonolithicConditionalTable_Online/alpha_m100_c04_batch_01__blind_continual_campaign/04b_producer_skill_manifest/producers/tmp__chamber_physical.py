exec(open('/tmp/chamber_probe.py').read().split('# Regress each component')[0])
from scipy.optimize import least_squares
L=q['lab_id']; S=q['stratum']; xq=s._features(q,q)[:len(h)]; D=xq@w[3:]+w[2]; O=np.ones(len(h)); miss=~np.isfinite(q['coupon_features_raw']); clean=~miss.any(axis=1); ids=np.arange(len(h)); C=np.c_[O,h,D];
# Subtract exact hold amplitudes and profile the remaining coupling.
def predict(z,ii,kind):
 B=basis(z); base=h[:,None]*(B[:,:2]@w[:2])[None,:] if 'hold' in kind else np.zeros_like(y)
 if 'hold' not in kind: G=[C,C,C]
 elif 'pure' in kind: G=[D[:,None],D[:,None],np.empty((len(h),0))]
 elif 'intercept' in kind: G=[np.c_[O,D],np.c_[O,D],np.empty((len(h),0))]
 else: G=[np.c_[O,D],np.c_[O,D],np.c_[O,D]]
 F=np.concatenate([B[None,:,j,None]*J[:,None,:] for j,J in enumerate(G)],axis=-1)
 b=np.linalg.lstsq(F[ii].reshape(-1,F.shape[-1]),(y-base)[ii].ravel(),rcond=None)[0]; return base+F@b,b
for kind in ('hold_pure','hold_intercept','hold_drift','free'):
 for free in (False,True):
  def fit(ii):
   z=least_squares(lambda z:(predict(z,ii,kind)[0][ii]-y[ii]).ravel(),tau,bounds=([.5,2.5],[1,10]),max_nfev=70).x if free else tau
   f,b=predict(z,ii,kind); return f,z,b
  f,z,b=fit(ids[clean]); E=[]
  for groups in (ids,L,S):
   err=np.zeros_like(y)
   for j in np.unique(groups):
    m=groups==j; err[m]=(fit(ids[~m&clean])[0]-y)[m]
   E.append([np.max(np.sqrt(np.mean(err[clean,:-1]**2,axis=0))),np.sqrt(np.mean(err[clean,-1]**2))])
  print(kind,free,'tau',z.round(6),'coef',b.round(6),'clean fit RMS',np.sqrt(np.mean((f-y)[clean]**2)).round(6),'clean CV',np.round(E,5),'missing signed end/rate',(f-y)[~clean,-2:].round(5))
# Determine whether the missing observations can be explained by one fixed imputation value per feature.
B=basis(tau); kind='hold_drift'; f,b=predict(tau,ids[clean],kind); direct=B@np.array([b[1],b[3],b[5]]); corr=(y-f)@direct/(direct@direct); print('missing inferred standardized values', (xq+corr[:,None]/w[3:])[miss].round(6)); print('diagnostic imputed',xq[miss].round(6),'deployment imputation',mean.round(6)); print('range check diagnostic and test D',np.percentile(D,[0,50,100]),np.percentile(xx[n:]@w[3:]+w[2],[0,50,100])); print('fit clean remaining shape rms',np.sqrt(np.mean((f-y+corr[:,None]*direct)[clean]**2,axis=0)).round(5))
