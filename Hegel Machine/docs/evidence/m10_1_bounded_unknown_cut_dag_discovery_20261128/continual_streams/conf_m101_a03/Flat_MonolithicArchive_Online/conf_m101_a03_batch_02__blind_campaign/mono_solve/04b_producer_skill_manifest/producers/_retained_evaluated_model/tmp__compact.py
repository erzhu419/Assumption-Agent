from measure import *
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
z=z[:n]; p=s['panel_c']; u=p['u']; V=p['v1']; A=lambda q:np.c_[np.ones(len(q)),q]; Q=lambda q:np.column_stack([A(q)]+[q[:,i]*q[:,j] for i,j in combinations_with_replacement(range(4),2)])
for fixed in [lambda q:np.ones((len(q),1)), A, lambda q:np.c_[A(q),q*q], Q]:
 for func in [np.tanh,np.sin,lambda a:a/np.sqrt(1+a*a)]:
  def features(q,c): return np.c_[fixed(q),func(q*np.exp(c[:4])+c[4:])]
  def resid(c):
   f=features(u,c); return (f@m._fit(f,V)-V).ravel()
  r=least_squares(resid,np.zeros(8),bounds=(np.r_[[-2]*4,[-2]*4],np.r_[[2]*4,[2]*4]),max_nfev=120)
  f=features(u,r.x); h=features(z,r.x)@m._fit(f,V)
  cv=evaluate(np.c_[Q(z),h]); print('C1',fixed(u).shape[1],func.__name__,np.std(r.fun),r.x,'y/traj required and worst',cv[0,:2],cv.max(0))
T=Y[:,1:]; _,_,vt=np.linalg.svd(T[:52],full_matrices=False); f=m._basis(z); print('residual-y vs trajectory',np.corrcoef((Y-f@m._fit(f[:52],Y[:52])).T)[0]); print('required/worst scores:')
for name,g in [('base',f),('no-square',np.c_[A(z),np.tanh(z)]),('sparse-square',np.c_[A(z),np.tanh(z),z[:,0]*z[:,3]]),('tanh focus',np.c_[Q(z),np.tanh(z[:,[1,3]])]),('tanh scaled .5',np.c_[Q(z),np.tanh(.5*z)]),('tanh scaled 1.5',np.c_[Q(z),np.tanh(1.5*z)])]:
 ans=evaluate(g); print(name,ans[0],ans.max(0))
for alpha in [.001,.01,.1,1]:
 out=[]
 for a,b in splits:
  G=np.c_[Q(z),np.tanh(z)-z]; G=G/G[a].std(0,where=np.broadcast_to(np.arange(G.shape[1])!=0,G[a].shape)).clip(1e-8) if False else G
  H=np.diag(np.r_[np.zeros(5),np.ones(10),np.full(4,4.)]); beta=np.linalg.solve(G[a].T@G[a]+alpha*H,G[a].T@Y[a]); pr=G[b]@beta; vb=np.linalg.svd(T[a],full_matrices=False)[2][:2]; pr[:,1:]=pr[:,1:]@vb.T@vb; e=np.abs(Y[b]-pr); out.append([e[:,0].mean(),e[:,1:].mean(),e[:,-1].mean(),e[:,1:].mean(0).max()])
 out=np.array(out); print('ridge + rank2',alpha,out[0],out.max(0))
