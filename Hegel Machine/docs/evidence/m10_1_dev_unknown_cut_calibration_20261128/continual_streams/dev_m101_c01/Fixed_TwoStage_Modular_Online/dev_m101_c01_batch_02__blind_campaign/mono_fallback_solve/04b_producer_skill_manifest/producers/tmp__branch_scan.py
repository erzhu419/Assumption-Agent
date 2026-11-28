import pickle,itertools as it,numpy as np
from scipy.optimize import least_squares
from scipy.linalg import solve
from scipy.spatial.distance import cdist
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; d=s['train']; x=d['x']; Y=np.c_[d['y'],d['traj']]; n=len(x)
def a(q):return np.column_stack((np.ones(len(q)),q))
def poly(q):return np.c_[a(q),*[q[:,i]*q[:,j] for i,j in it.combinations_with_replacement(range(q.shape[1]),2)]]
p=s['panel_a']; u,v=p['u'],p['v']; ii=np.ones(len(u),bool)
for _ in range(5):
 A=np.linalg.lstsq(a(u[ii]),v[ii],rcond=None)[0]; r=np.linalg.norm(a(u)@A-v,axis=1); ii=r<4*np.median(r)
A[1:]=np.diag(np.diag(A[1:])); p=s['panel_b']; B=np.linalg.lstsq(a(p['u']),p['v'],rcond=None)[0]; p=s['panel_c']; u=p['u']
def fit_c(key, f0, warp, bounds):
 v=p[key]
 def f(p,q):return a(warp(p,q))
 def res(p):
  m=f(p,u);return (m@np.linalg.lstsq(m,v,rcond=None)[0]-v).ravel()
 fit=least_squares(res,f0,bounds=bounds,ftol=1e-10,gtol=1e-10,xtol=1e-10); C=np.linalg.lstsq(f(fit.x,u),v,rcond=None)[0]; print(key,fit.x,np.sqrt(np.mean(fit.fun**2)),flush=True)
 return lambda q: f(fit.x,q)@C, lambda q:warp(fit.x,q)
C1,W1=fit_c('v1',[2.25,.45,0,0],lambda p,q:np.tanh(p[0]*q+p[2])+p[1]*q+p[3]*(1-np.cos(2.5*q)),([.2,-1,-1,-1],[5,2,1,1]))
C2,W2=fit_c('v2',[1.3,.65,.02],lambda p,q:q+p[1]*(1-np.cos(p[0]*q))+p[2]*q*q,([.2,-2,-2],[5,2,2]))
print('varying parameters:'); C1f,W1f=fit_c('v1',[2.25]*4+[.45,0],lambda p,q:np.tanh(q*p[:4]+p[5])+p[4]*q,([.2]*4+[-1,-1],[5]*4+[2,1])); C2f,W2f=fit_c('v2',[2.5,.2,.1],lambda p,q:q+p[1]*(1-np.cos(p[0]*q))+p[2]*q*q,([1.4,-2,-2],[5,2,2]))
from scipy.linalg import eigh
rng=np.random.default_rng(22); folds=[np.arange(56,72),*np.array_split(rng.permutation(n),5)]
def errors(K,lam):
 r=[]
 for j in folds:
  i=np.setdiff1d(np.arange(n),j); e=np.abs(K[np.ix_(j,i)]@solve(K[np.ix_(i,i)]+lam*np.eye(len(i)),Y[i],assume_a='pos')-Y[j]);r.append([e[:,0].mean(),e[:,1:].mean()])
 return np.array(r)
ops={'a':lambda q:(q-A[0])/np.diag(A[1:]),'A':lambda q:a(q)@A,'B':lambda q:a(q)@B,'b':lambda q:(q-B[0])@np.linalg.inv(B[1:])}
frames={'x':x,**{k:f(x) for k,f in ops.items()}}
frames.update({k+l:g(f(x)) for k,f in ops.items() for l,g in ops.items() if k.lower()!=l.lower()})
res=[]
for name,q in frames.items():
 P=poly(q); Q=np.linalg.qr(P)[0]; D=Y-Q@(Q.T@Y)
 candidates={'C1':C1(q),'C2':C2(q),'W1':W1(q),'W2':W2(q),'W1W2':np.c_[W1(q),W2(q)]}
 for lab,g in list(candidates.items())[:4]:
  candidates[lab+'x']=np.einsum('ni,nj->nij',q,g).reshape(n,-1)
  for alpha in [-1,-.5,.3,.6,1,2]: candidates[lab+str(alpha)]=poly(q+alpha*g)
 for lab,g in candidates.items():
  g=g-Q@(Q.T@g); scale=np.linalg.norm(g)/np.sqrt(n)
  if scale<1e-12:continue
  Gn=g/scale
  for lam in [.003,.03,.3]:
   r=errors(P@P.T+Gn@Gn.T,lam);res.append((name,lab,lam,r))
for k in range(2):
 print('top branch fits',k)
 for name,lab,lam,r in sorted(res,key=lambda r:r[-1][:,k].mean())[:8]:print(name,lab,lam,np.round(r[:,k],5))
_,S,V=np.linalg.svd(Y[:,1:],full_matrices=False); Z=Y[:,1:]@V[:2].T; print('y/amp corr',np.corrcoef(np.c_[Y[:,0],Z].T));
for label,Q in [('amp',poly(Z)),('x+amp',poly(np.c_[x,Z])),('xquad+amp',np.c_[poly(x),Z]),('x+ampquad',np.c_[a(x),poly(Z)]),('xquad',poly(x))]:
 r=errors(Q@Q.T,1.e-4);print(label,'observed amp scalar MAE',np.round(r[:,0],5))
# Fit temporal modes with candidate physical functions; measures representational floor.
t=s['eval_times']; z=Y[:,1:]; exp=np.exp
F={'t':lambda p:t,'t2':lambda p:t*t,'relax':lambda p:-np.expm1(-p[2]*t),'sind':lambda p:exp(-p[0]*t)*np.sin(p[1]*t),'cosd0':lambda p:1-exp(-p[0]*t)*np.cos(p[1]*t),'saturate_sin':lambda p:(1-exp(-p[0]*t))*np.sin(p[1]*t),'log':lambda p:np.log1p(p[2]*t),'td':lambda p:t*exp(-p[2]*t),'rampcos':lambda p:t*(1-np.cos(p[2]*t))}
for a1 in ['sind','cosd0','saturate_sin']:
 for a2 in F:
  if a1==a2:continue
  def fun(p):
   P=np.column_stack((F[a1](p),F[a2](p)));return (P@np.linalg.lstsq(P,z.T,rcond=None)[0]-z.T).ravel()
  fit=least_squares(fun,[.2,1.2,.2],bounds=([0,.1,.0001],[5,6,5]),max_nfev=100); ma=np.mean(abs(fit.fun));
  if ma<.0045:print(a1,a2,np.round(fit.x,6),round(ma,6))
