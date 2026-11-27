import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
    exec(open('/tmp/m98_structure.py').read(),globals())
from scipy.optimize import minimize_scalar
T=np.r_[t,0.]; R=np.column_stack((np.r_[-np.expm1(-t/tau[0]),1/tau[0]],np.r_[-np.expm1(-t/tau[1]),1/tau[1]])); Y0=Y-h[:,None]*(R@w[:2]);
print('hold-only residual matrix singular values:',np.linalg.svd(Y0,compute_uv=False).round(5)); print('residual vs nominal drift:',np.linalg.lstsq(D,Y0,rcond=None)[0].round(5)); print('one-variable residual RMSE:',np.sqrt(np.mean((Y0-D@np.linalg.lstsq(D,Y0,rcond=None)[0])**2,axis=0)).round(5))
# Keep the required train+test normalization fixed, while testing whether chamber-panel normalization changes missing-value predictions.
raw=dg['coupon_features_raw']; print('diagnostics missing',np.isfinite(raw).sum(0),np.any(~np.isfinite(raw),1).nonzero()[0]);
for lab in np.unique(tr['lab_id']):
 print('panel',lab,[(round(np.median(a.mean(0)),4),round(np.median(a.std(0)),4)) for g0 in (tr,te,dg) for a in [g0['ref_coupons'][g0['lab_id']==lab]]])

def modes(q):
 e=np.exp(-t/q)
 return [np.r_[-np.expm1(-t/q),1/q],np.r_[t*e,1.],np.r_[t*e*0+q*(-np.expm1(-t/q)),1.],np.r_[t/(1+t/q),1.],np.r_[t/(1+(t/q)**2),1.],np.r_[q*np.log1p(t/q),1.],np.r_[t-q*(-np.expm1(-t/q)),0.]]

def estimate(B,ix,iy=None,target=Y0):
 if iy is None:iy=np.arange(len(Y))
 C=np.linalg.lstsq(B[ix].reshape(-1,B.shape[-1]),target[ix].ravel(),rcond=None)[0]; return B[iy]@C,C

def scores(E):return np.array([np.sqrt(np.mean(E[:,:K]**2)),np.sqrt(np.mean(E[:,K]**2)),max(np.sqrt(np.mean(E[:,:K]**2,axis=1)))])
F=[np.flatnonzero(np.arange(n)!=j) for j in range(n)]
block=lambda z,b: z[:,None,:]*np.asarray(b)[None,:,None]
for off in (False,True):
 for gkind in ('D','DX','D2'):
  G0={'D':D,'DX':X,'D2':np.c_[D,D**2]}[gkind]
  for mode in [0,1,3,4,5,6]:
   def build(q):
    B=block(G0,modes(q)[mode]);
    return np.concatenate((B,block(np.c_[H,one],R[:,0]),block(np.c_[H,one],R[:,1])),2) if off else B
   def fit(ix):
    def loss(lq):
     E=estimate(build(np.exp(lq)),ix,ix)[0]-Y0[ix];return np.sum(E**2)
    q=np.exp(minimize_scalar(loss,bounds=(-3.,5.), method='bounded').x);return q,build(q)
   q,B=fit(np.arange(n)); pred,C=estimate(B,np.arange(n));
   cv=np.vstack([estimate(fit(ix)[1],ix,[j])[0]-Y0[j] for j,ix in enumerate(F)])
   print('drift mode',off,gkind,mode,round(q,4),'fit/cv',scores(pred-Y0).round(5),scores(cv).round(5),'coefs',C.round(4))
# Probe whether the remaining variations align with initial rate only, stress nonlinearity, or an altered memory kernel.
Z=np.c_[H,D,X[:,1:],p,p**2,p[:,1:]-p[:,:-1],p[:,1:]*p[:,:-1],h[:,None]*p,np.tanh(p),p/(1+p**2),h[:,None]**2]
q=2.5
E=Y0-D@modes(q)[2][None,:]
for y0 in (E[:,0],E[:,K],E[:,K-1]):
 cs=np.array([np.corrcoef(z,y0)[0,1] for z in Z.T]); inds=np.argsort(-abs(cs))[:10]; print('residual correlates (column,corr):',list(zip(inds,cs[inds].round(3))))
