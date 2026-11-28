from explore import *
from itertools import combinations_with_replacement
B=bv; U=u; C=np.sum(U*np.roll(B,1,axis=1),axis=1); N=len(y); i=np.arange(N)
W=np.c_[np.ones(N),B,C]; norm=np.sum(B*B,axis=1)
Q=np.column_stack([B[:,j]*B[:,k] for j,k in combinations_with_replacement(range(4),2)])
signed=(B@np.array([1.,-1.,-1.,1.])/2)**2
rng=np.random.default_rng(2027)
folds=[(i[:52],i[52:])]+[(p[:52],p[52:]) for p in (rng.permutation(N) for _ in range(40))]
for z in [*U.T,abs(C),np.linalg.norm(U,axis=1),np.linalg.norm(B,axis=1)]:
 j=np.argsort(z); folds.append((j[:-15],j[-15:]))
models={'base':np.empty((N,0)), 'signed-one':(norm-2*signed)[:,None], 'signed-two':np.c_[norm,signed], 'full-ridge':Q, 'isotropic-rank1':norm[:,None]}
for name, X in models.items():
 E=[]
 for I,J in folds:
  w=np.linalg.lstsq(W[I],y[I],rcond=None)[0]; m=X.shape[1]
  p0=np.r_[2.,w[1:5]/2,0.,w[5],w[0],np.zeros(m)]
  if name=='isotropic-rank1':p0=np.r_[p0,.25*np.array([1.,-1.,-1.,1.])]
  def pred(p,K):
   z=p[0]*np.tanh(B[K]@p[1:5]+p[5])+p[6]*C[K]+p[7]+X[K]@p[8:8+m]
   return z-(B[K]@p[9:])**2 if name=='isotropic-rank1' else z
  def err(p):
   e=pred(p,I)-y[I]
   return np.r_[e,np.sqrt(.3)*X[I].std(0)*p[8:]] if name=='full-ridge' else e
  fit=least_squares(err,p0,max_nfev=250)
  E.append(np.mean(abs(pred(fit.x,J)-y[J])))
  if len(E)==1: print(name,'public-fit coefficients',np.round(fit.x,5))
 print(name,'public/mean-random/worst-random/worst-tail',np.round([E[0],np.mean(E[1:41]),max(E[1:41]),max(E[41:])],6))
for name,X in [('base',W),('base+coords',np.c_[W,U]),('base+coords+signed',np.c_[W,U,norm,signed])]:
 E=np.array([np.mean(abs(X[J]@np.linalg.lstsq(X[I],tr[I],rcond=None)[0]-tr[J]),axis=0) for I,J in folds]); e=E.mean(axis=1)
 print('trajectory',name,'public/worst-random/worst-tail/worst-horizon',np.round([e[0],max(e[1:41]),max(e[41:]),E.max()],6))
