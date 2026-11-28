exec(open('/tmp/chamber_law.py').read().split('for label,F in')[0])
from scipy.optimize import least_squares
lab=c['lab_id']; missing=np.isnan(c['coupon_features_raw']).any(axis=1); idx=np.arange(n); g=theta[2]+xc@theta[3:]; G=np.c_[np.ones(n),g]; H=np.c_[hc,g]; F=np.c_[np.ones(n),hc,g]; Z=F[:,:0]
# Fixed physical hold-load slope with editable coupling; compare free slopes,
# affine drift, and how much fitting only complete rows changes predictions.
for kind in ['anchor','coupling','coupling_drift','affine','affine_drift']:
 for free in [0,1,2]:
  def fit(ix):
   features={'anchor':(Z,Z,Z),'coupling':(H[:,1:],H[:,1:],Z),'coupling_drift':(H[:,1:],H[:,1:],G),'affine':(F,F,Z),'affine_drift':(F,F,G)}[kind]
   if kind=='anchor': off=np.c_[theta[0]*hc+.3*g,theta[1]*hc+2*g,np.zeros(n)]
   elif kind.startswith('coupling'): off=np.c_[theta[0]*hc,theta[1]*hc,np.zeros(n)]
   else: off=np.zeros((n,3))
   k=sum(f.shape[1] for f in features)
   def form(p):
    delta=np.zeros(2); delta[:free]=p[k:]; tt=tau*np.exp(delta)
    B=np.vstack((np.c_[-np.expm1(-t[:,None]/tt),t],.5*np.r_[1/tt,1]))
    A=np.concatenate([f[:,None,:]*B[None,:,j,None] for j,f in enumerate(features)],axis=2)
    return A,off@B.T,tt
   p0=np.zeros(k+free); A,O,_=form(p0)
   if k:p0[:k]=np.linalg.lstsq(A[ix].reshape(-1,k),(Y-O)[ix].ravel(),rcond=None)[0]
   def res(p):
    A,O,_=form(p); return (A[ix]@p[:k]+O[ix]-Y[ix]).ravel()
   p=least_squares(res,p0,loss='cauchy',f_scale=.004,x_scale='jac',max_nfev=110).x if len(p0) else p0
   A,O,tt=form(p); return A@p[:k]+O-Y,tt,p
  e,tt,p=fit(idx); cv=e.copy(); comp=e.copy()
  for l in np.unique(lab):
   mask=lab==l; cv[mask]=fit(idx[~mask])[0][mask]; comp[mask]=fit(idx[~mask & ~missing])[0][mask]
  print(kind,free,'all',report(cv),'complete',report(cv[~missing]),'complete-fit all',report(comp),'t',tt.round(6),'p',p.round(6))
