import numpy as np
from scipy.optimize import least_squares
z=np.load('/tmp/chamber_params.npz'); B,y,p,h,f,x,a,lab,tau,th=[z[k] for k in ('B','y','p','h','f','x','a','lab','tau','th')]; reset=z['reset'].copy(); reset[:,2]=0; Y=y-reset@B.T
print('force and inferred response ratio',np.c_[h,f,np.divide(a-reset,f[:,None])].round(3)); print('correction singular values',np.linalg.svd(a-reset,compute_uv=False))
ones=np.ones_like(f); fs=f[:,None]
features={'linear':fs,'shift':np.c_[ones,f],'poly2':np.c_[f,f*f],'poly3':np.c_[f,f*f,f**3],'x':np.c_[ones,x],'h*f':np.c_[f,f*h],'h^2*f':np.c_[f,f*h,f*h*h],'x*f':np.c_[f,fs*x], 'p*f':np.c_[f,fs*p], 'poly3+hf':np.c_[f,f*f,f**3,f*h]}
for kind,F in features.items():
 pred=np.zeros_like(Y); cs=[]
 for l in [*np.unique(lab),-1]:
  sel=lab!=l
  def err(c): return (F[sel]@c.reshape(-1,3)@B.T-Y[sel]).ravel()
  c=least_squares(err,np.linalg.lstsq(F[sel],(a-reset)[sel],rcond=None)[0].ravel(),loss='soft_l1',f_scale=.008).x.reshape(-1,3)
  pred[~sel]=F[~sel]@c@B.T
  if l==-1:print(kind,'coeff',c.round(4).tolist())
 e=pred-Y; print('worst curve/rate by lab',np.max([np.sqrt(np.mean(e[lab==l,:-1]**2)),] for l in []) if False else np.max([np.sqrt(np.mean(e[lab==l,:-1]**2)) for l in np.unique(lab)]),np.max([np.sqrt(np.mean(e[lab==l,-1]**2)) for l in np.unique(lab)]))
