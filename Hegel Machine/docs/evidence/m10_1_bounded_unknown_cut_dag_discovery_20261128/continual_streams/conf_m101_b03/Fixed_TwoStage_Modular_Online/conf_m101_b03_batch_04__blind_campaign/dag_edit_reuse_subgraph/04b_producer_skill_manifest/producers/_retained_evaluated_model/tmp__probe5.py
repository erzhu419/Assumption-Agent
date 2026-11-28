from p import *
from scipy.optimize import minimize, minimize_scalar
from scipy.linalg import qr
np.set_printoptions(precision=5,suppress=True,linewidth=160)
U,S,V=np.linalg.svd(T,full_matrices=False)
for name,gen,start in [('step+t',lambda p:np.c_[-np.expm1(-p[0]*t),t],[1.]),('step+te',lambda p:np.c_[-np.expm1(-p[0]*t),t*np.exp(-p[0]*t)],[1.]),('two steps',lambda p:np.column_stack([-np.expm1(-q*t) for q in p]),[.3,1.4]),('step+sine',lambda p:np.c_[-np.expm1(-p[0]*t),np.sin(p[1]*t)],[1.,.7])]:
 def loss(p):
  A=gen(p); return np.mean((T-T@np.linalg.pinv(A).T@A.T)**2)
 opt=minimize(loss,start,method='Nelder-Mead',options={'xatol':1e-10}); A=gen(opt.x); print(name,opt.x,np.sqrt(opt.fun),A.T)
 print('amplitudes ranges',np.ptp(T@np.linalg.pinv(A).T,axis=0))
print('SVD basis',V[:2]); print('noise floor',np.sqrt(np.sum(S[2:]**2)/T.size))
Z=np.c_[y,U[:,:2]*S[:2]]
for k in (1,2,3):
 A=poly(k); E=Z-A@np.linalg.lstsq(A,Z,rcond=None)[0]; _,D,W=np.linalg.svd(E,full_matrices=False); print('residual basis',k,D,W)
 if k==2:
  C=np.linalg.lstsq(A,Z@W.T,rcond=None)[0]; print('poly2 target rotated coefficients',C)
for name in ('panel_b','panel_c'):
 a=s[name]['u']; ids=[c for k in range(3) for c in combinations_with_replacement(range(4),k)]; A=np.column_stack([np.prod(a[:,c],axis=1) for c in ids]); print(name)
 for key,b in s[name].items():
  if key=='u': continue
  C=np.linalg.lstsq(A,b,rcond=None)[0]; print(key,'quadratic coefficients\n',C,'rmse',np.sqrt(np.mean((b-A@C)**2,axis=0)))
# See if a common rational divisor reduces cubic residuals without a large model.
for kind in ('squares','quadratic','diagonal_affine'):
 A=poly(3); aa=poly(2); B=x*x if kind=='squares' else aa[:,5:] if kind=='quadratic' else np.c_[x,x*x];
 for scale in (0.,.01,.1):
  errs=[]
  for jj in [np.arange(52,72),*np.array_split(np.random.default_rng(39).permutation(72),6)]:
   ii=np.setdiff1d(np.arange(72),jj); F=A[ii]; q=np.linalg.qr(F,mode='reduced')[0]; yi=Z[ii]; H=(yi[:,:,None]*B[ii,None,:]).reshape(len(ii),-1); E=yi-q@(q.T@yi); H=(H-q@(q.T@H)).reshape(len(ii),3,-1); C=np.linalg.lstsq(np.vstack([H.reshape(-1,B.shape[1]),np.eye(B.shape[1])*scale]),np.r_[-E.ravel(),np.zeros(B.shape[1])],rcond=None)[0]; den=1+B@C; P=np.linalg.lstsq(F,yi*den[ii,None],rcond=None)[0]; pred=(A[jj]@P)/den[jj,None]; errs.append([np.mean(abs(pred[:,0]-y[jj])),np.mean(abs(pred[:,1:]@V[:2]-T[jj])),den.min()])
  print('rational',kind,scale,'tail',np.round(errs[0],4),'worst',np.max(errs,0).round(4))
