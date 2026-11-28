exec(open('/tmp/explore.py').read().split('a=s[')[0])
from scipy.linalg import cho_factor,cho_solve
from scipy.special import expit
p=s['panel_a']; g=np.linalg.norm(p['v']-p['u'],axis=1)<2; A=ls(poly(p['u'][g],1),p['v'][g]); X=(x-A[0])@np.linalg.inv(A[1:]); P=poly(X,2); R=poly(X,1)
U,S,V=np.linalg.svd(z,full_matrices=False); Y=np.c_[y,U[:,0]*S[0],U[:,1]*S[1]]; Q=-np.expm1(-t[:,None]*np.array([1.6,.2])); print('SVD',V[:2],S[:3]); print('time basis residual',np.std(z-ls(Q,z.T).T@Q.T), 'projection vals',ls(Q,V[:2].T));
rng=np.random.default_rng(129); folds=[np.arange(56,72)]+np.array_split(rng.permutation(n),5)+np.array_split(rng.permutation(n),5)
for j in range(2):
 out=Y[:,j]; q=ls(P,out); print('output',j,'poly/lin predictions','curve corr',np.corrcoef(np.c_[out-P@q,(P@q)**3, np.tanh(P@q), z[:,-1], (z[:,-1])**2].T)[0]);
 def predict(nm,it,iv):
  T=P[it]; yy=out[it]; coef=ls(T,yy); p0=coef
  if nm=='quad':return P[iv]@coef
  if nm.startswith('link'):
   if nm=='link_asinh':fun=lambda q,r:np.arcsinh(q*r)/r
   if nm=='link_tanh':fun=lambda q,r:np.tanh(q*r)/r
   if nm=='link_sin':fun=lambda q,r:np.sin(q*r)/r
   if nm=='link_rat':fun=lambda q,r:q/(1+r*r*q*q)
   fit=least_squares(lambda b:fun(T@b[:-1],b[-1])-yy,np.r_[coef,.1],bounds=(np.r_[np.full(T.shape[1],-np.inf),.001],np.r_[np.full(T.shape[1],np.inf),1.]),max_nfev=100)
   return fun(P[iv]@fit.x[:-1],fit.x[-1])
  def mats(ix,r):
   v=R[ix]@r
   if nm=='tanhridge':f=np.tanh(v)
   if nm=='sinridge':f=np.sin(v)
   if nm in ('ratridge','ratlinear','ratquad'):f=1/(1+v*v)
   if nm in ('tanhridge','sinridge','ratridge'):return np.c_[P[ix],f]
   if nm=='ratlinear': return np.c_[P[ix],f[:,None]*R[ix]]
   if nm=='ratquad': return np.c_[P[ix],f[:,None]*P[ix]]
  df=2 if nm=='ratquad' else 5
  starts=np.r_[np.eye(5),np.random.default_rng(1).normal(size=(df,5))*.5]; starts[0]=[.2,.4,.4,.4,.4]
  def residual(r):
   F=mats(it,r); c=ls(F,yy); return np.r_[F@c-yy,r*.002,c*.0005]
  fits=[least_squares(residual,r,max_nfev=70,bounds=(-5,5)) for r in starts]; fit=min(fits,key=lambda v:v.cost); F=mats(it,fit.x); c=ls(F,yy)
  if len(iv)==0: print('fit',nm,round(np.sqrt(2*fit.cost/len(it)),5),fit.x,c[-5:]);return []
  return mats(iv,fit.x)@c
 for nm in ['quad','link_asinh','link_tanh','link_sin','link_rat','tanhridge','sinridge','ratridge','ratlinear','ratquad']:
  if not(nm.startswith('link') or nm=='quad'):predict(nm,np.arange(n),np.arange(0))
  E=[]
  for iv in folds:
   it=np.setdiff1d(np.arange(n),iv); E.append(np.mean(abs(predict(nm,it,iv)-out[iv])))
  print(nm,'56/16, mean, max',E[0],np.mean(E),np.max(E),flush=True)
