exec(open('/tmp/probe_chamber.py').read().split('for j,xq in enumerate(xqs):')[0])
from scipy.optimize import least_squares
P,H=g(tr); X=f(xtr); O=y(tr); Q=d['chamber_verification_trials']; missing=np.isnan(Q['coupon_features_raw']).any(1)
def residual(a):
 c,r=m._design(a[:2],P,H,X,dt,t)
 return np.column_stack((c@a[2:],r@a[2:]))-O
x=np.r_[taus,th]; e=abs(residual(x)); print('residual gap',np.round(np.sort(e[:,:-1],axis=None)[-70::3],4));
for limit in [.04,.06,.10]:
 w=np.where(np.arange(O.shape[1])==O.shape[1]-1,2.,1.); keep=e<limit
 a=least_squares(lambda a:(residual(a)*w)[keep],x,x_scale='jac').x
 pred=Y-np.column_stack((h[:,None]*(-np.expm1(-t[:,None]/a[:2])@a[2:4]), h*(a[2:4]@(1/a[:2]))))
 dr=f(xqs[-1])@a[4:]; F=-np.expm1(-t[:,None]/a[:2])*a[:2]-t[:,None]
 print('refit',limit,np.round(a,7),'diagnostic complete initial rate RMS',np.sqrt(np.mean((pred[~missing,-1]-dr[~missing])**2)))
 for weight in [None,[.175,.55],[.18,.54],[.175,.545]]:
  rows=~missing; A=dr[:,None,None]*F[None,:,:]; target=pred[:,:-1]-dr[:,None]*t
  b=np.linalg.lstsq(A[rows].reshape(-1,2),target[rows].ravel(),rcond=None)[0] if weight is None else weight
  E=target-A@b
  print('kernel',np.round(b,6),'complete RMSE',np.sqrt(np.mean(E[rows]**2)),'late bias',E[rows,-1].mean())
 print('missing measured minus prescribed rate drift',np.round((pred[:,-1]-dr)[missing],5))
