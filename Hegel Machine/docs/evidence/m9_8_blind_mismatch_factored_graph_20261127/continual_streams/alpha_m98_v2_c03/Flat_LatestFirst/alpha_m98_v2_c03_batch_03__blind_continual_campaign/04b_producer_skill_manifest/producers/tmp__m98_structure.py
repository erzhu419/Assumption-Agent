import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 from m98_explore import *
from scipy.optimize import least_squares
Y=y(dg); past,hold,X=diag; z=hold[:,None]-a[:,-1,:2]*taus
S=np.c_[hold,past]; U=np.c_[np.ones(len(X)),S,X[:,1:]]; T=np.r_[t,0.]; Z=np.linalg.lstsq(U,Y,rcond=None)[0]; print('singular values',np.linalg.svd(Y,compute_uv=False).round(4)); print('stress and feature OLS profiles (columns first/middle/end/rate)'); print(Z[:,[0,4,8,11,12]].round(4)); print('feature rank singular values',np.linalg.svd(Z[-4:],compute_uv=False).round(4)); print('implied drift profile if same direction',((th[3:]@Z[-4:])/(th[3:]@th[3:])).round(4)); print('feature profiles divided by train weights');print((Z[-4:]/th[3:,None]).round(4)); print('stress profiles expressed in baseline modes'); print(np.linalg.lstsq(F[:,:2],Z[1:8].T,rcond=None)[0].T.round(4)); print('sweep keys',[(k,np.shape(v)) for k,v in cal['single_channel_sweeps'].items()]);
# Allow simple chamber rescaling of history and drift, and compare alternative time laws.
r=X@th[2:]; E=F[:,:2]; w=np.r_[np.ones(len(t)),1.4]
def drift_curve(kind,p):
 if kind=='linear':return np.r_[t,1.]
 if kind=='quad':return np.r_[t+p*t*t,1.]
 if kind=='saturating':return np.r_[-np.expm1(-p*t)/p,1.]
 if kind=='rational':return np.r_[t/(1+p*t),1.]
 if kind=='delayed':return np.r_[t+np.expm1(-p*t)/p,0.]
 if kind=='hybrid':return np.r_[t+p[0]*(-np.expm1(-p[1]*t))/p[1],1+p[0]]
def design(ix,kind,p,drive,offset):
 terms=[hold[:,None]*E[:,0]*th[0],hold[:,None]*E[:,1]*th[1],z[:,0,None]*E[:,0]*th[0],z[:,1,None]*E[:,1]*th[1]]
 if offset:terms += [np.broadcast_to(q,Y.shape) for q in E.T]
 terms += [u[:,None]*drift_curve(kind,p) for u in drive.T];return np.stack(terms,-1)[ix]
for name in ['linear','quad','saturating','rational','delayed','hybrid']:
 for label,drive in [('intrinsic',r[:,None]),('affine',np.c_[np.ones(len(r)),r]),('separate',X)]:
  for offset in (False,True):
   def fit(ix):
    def linear(p):
     q=design(ix,name,p,drive,offset)*w[None,:,None];return np.linalg.lstsq(q.reshape(-1,q.shape[-1]),(Y[ix]*w).ravel(),rcond=None)[0]
    def obj(p):return ((design(ix,name,p,drive,offset)@linear(p)-Y[ix])*w).ravel()
    p=[] if name=='linear' else least_squares(obj,[0.25] if name!='hybrid' else [-0.3,0.5],bounds=([-0.1],[5]) if name in ['quad'] else (([-10,0.003],[10,15]) if name=='hybrid' else ([0.003],[15]))).x
    return p,linear(p)
   idx=np.arange(len(r)); p,c=fit(idx); res=design(idx,name,p,drive,offset)@c-Y; cv=[]
   for lab in np.unique(dg['lab_id']):
    ix=idx[dg['lab_id']!=lab]; j=idx[dg['lab_id']==lab]; pp,cc=fit(ix); u=design(j,name,pp,drive,offset)@cc-Y[j]; cv.append([np.sqrt(np.mean(u[:,:-1]**2)),np.sqrt(np.mean(u[:,-1]**2))]);
   print(name,label,offset,'fit',np.sqrt(np.mean(res[:,:-1]**2)).round(4),np.sqrt(np.mean(res[:,-1]**2)).round(4),'worst lab',np.max(cv,axis=0).round(4),'p',np.round(p,4),'coef',c.round(3))
