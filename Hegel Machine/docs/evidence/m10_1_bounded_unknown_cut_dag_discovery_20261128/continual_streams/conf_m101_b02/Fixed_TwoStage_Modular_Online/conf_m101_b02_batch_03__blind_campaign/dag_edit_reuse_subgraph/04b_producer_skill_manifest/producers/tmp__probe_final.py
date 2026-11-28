import numpy as np
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import minimize
from itertools import combinations,combinations_with_replacement
D=np.load('/tmp/readout_arrays.npz'); Z=D['z']; Y=D['Y']; n=len(Z); X=np.column_stack([D[k] for k in ['z','v','v1','v2']]); X=(X-X.mean(0))/X.std(0)
folds=[(np.arange(52),np.arange(52,n))]+[(np.setdiff1d(np.arange(n),b),b) for b in np.array_split(np.random.default_rng(13).permutation(n),4)]
def poly(z,d):return np.column_stack([np.ones(len(z))]+[np.prod(z[:,j],axis=1) for k in range(1,d+1) for j in combinations_with_replacement(range(z.shape[1]),k)])
def summary(es):return np.round([*es[0],*np.max(es[1:],axis=0),*np.mean(es[1:],axis=0)],5)
def evaluate(fn):
 es=[]
 for a,b in folds:
  V=np.linalg.svd(Y[a,1:],full_matrices=False)[2][:2]; T=np.c_[Y[:,0],Y[:,1:]@V.T]; P=fn(a,b,T); es.append([np.mean(abs(P[:,0]-Y[b,0])),np.mean(abs(P[:,1:]@V-Y[b,1:]))])
 return summary(es)
for group in [Z,X[:,4:8],X[:,:4],X[:,8:12],X[:,12:16]]:
 for deg in [2,3,4,5]:
  F=poly(group,deg)
  for l in [.0003,.01,.1,1.]:
   def f(a,b,T):
    A=F[a]; return F[b]@A.T@np.linalg.solve(A@A.T+l*np.eye(len(a)),T[a])
   q=evaluate(f)
   if max(q[:2])<.085:print('poly',np.argmax([group is Z,group.base is X]),deg,l,q,flush=True)

def gp(z,a,b,T,k,mean_deg):
 Q=poly(Z,mean_deg); Q=Q/(np.sqrt(np.mean(Q[a]**2,axis=0))+1e-9); out=[]
 diff=(z[:,None,:]-z[None,:,:])**2; da=diff[a][:,a]; db=diff[b][:,a]; d=z.shape[1]; G=Q[a]@Q[a].T; H=Q[b]@Q[a].T
 def ek(dd,h):
  L=np.exp(-2*h[:d]); r2=dd@L; r=np.sqrt(np.maximum(r2,1e-30)); s=np.exp(h[d]);
  if k=='rbf': A=np.exp(-r2/2); A1=-A/2
  elif k=='matern': A=(1+r+r2/3)*np.exp(-r); A1=-(1+r)*np.exp(-r)/6
  else:A=1/(1+r2/2); A1=-A*A/2
  return s*A,-2*s*A1[:,:,None]*dd*L
 for y in T.T:
  sd=y[a].std(); v=(y[a]-y[a].mean())/sd; I=np.eye(len(a))
  def obj(h):
   K,J=ek(da,h); g=np.exp(h[-2]); noise=np.exp(h[-1]); C=cho_factor(K+g*G+noise*I,check_finite=False); u=cho_solve(C,v,check_finite=False); M=(cho_solve(C,I,check_finite=False)-np.outer(u,u))/2; pen=.15*np.sum(h[:d]**2); return (v@u/2+np.log(np.diag(C[0])).sum()+pen,np.r_[np.einsum('ij,ijk->k',M,J)+.3*h[:d],np.sum(M*K),g*np.sum(M*G),noise*np.trace(M)])
  fits=[minimize(obj,np.r_[np.full(d,st),0.,-2.,-7.],jac=True,bounds=[(-2.5,2.8)]*d+[(-7,4),(-9,3),(-13,-1)],method='L-BFGS-B',options={'maxiter':110}) for st in [-.45,.65]]; h=min(fits,key=lambda p:p.fun).x
  W=ek(db,h)[0]+np.exp(h[-2])*H; K=ek(da,h)[0]+np.exp(h[-2])*G+np.exp(h[-1])*I; out.append(y[a].mean()+sd*(W@np.linalg.solve(K,v)))
 return np.column_stack(out)
for name,z in [('latent',Z),('B',X[:,4:8]),('C1',X[:,8:12]),('all',X),('latent+C',np.c_[Z,X[:,8:]])]:
 for mf in [1,2]:
  for k in ['rbf','matern','rq']:
   print('gp',name,mf,k,evaluate(lambda a,b,T:gp(z,a,b,T,k,mf)),flush=True)
