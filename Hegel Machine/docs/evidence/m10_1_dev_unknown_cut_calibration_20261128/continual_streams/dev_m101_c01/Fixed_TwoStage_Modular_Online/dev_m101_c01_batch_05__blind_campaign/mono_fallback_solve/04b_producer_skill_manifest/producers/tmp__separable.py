from spatial import *
from scipy.special import erf
from itertools import product
p=s['panel_c']; u=p['u'];
for k in ['v1','v2']:
 v=p[k]; forms={'tanh':lambda q,a:np.tanh(a*q),'erf':lambda q,a:erf(a*q),'atan':lambda q,a:np.arctan(a*q),'asinh':lambda q,a:np.arcsinh(a*q),'sin':lambda q,a:np.sin(a*q),'cos':lambda q,a:np.cos(a*q),'logcosh':lambda q,a:np.logaddexp(a*q,-a*q),'gauss':lambda q,a:np.exp(-(a*q)**2),'rat':lambda q,a:1/(1+(a*q)**2)}
 for nm,f in forms.items():
  am=np.linspace(.5,5,20)
  def model(a,q): return np.c_[np.ones(len(q)),q,f(q,a)]
  def res(a):
   M=model(a[0],u); return (M@L(M,v)-v).ravel()
  r=min([least_squares(res,[a],bounds=(.15,8)) for a in am[::4]],key=lambda r:r.cost)
  print(k,nm,r.x,np.std(r.fun))
 for nm,fn,ini in [('tanh+sin',lambda q,a:np.tanh(a[0]*q)+a[1]*np.sin(a[2]*q),[2.,.2,2.]),('erf+sin',lambda q,a:erf(a[0]*q)+a[1]*np.sin(a[2]*q),[1.5,.2,2.]),('tanh+atan',lambda q,a:np.tanh(a[0]*q)+a[1]*np.arctan(a[2]*q),[1.5,.2,3.]),('cos+cos',lambda q,a:np.cos(a[0]*q)+a[1]*np.cos(a[2]*q),[1.1,.01,4]),('cos+q2',lambda q,a:np.cos(a[0]*q)+a[1]*q*q,[2.,.2,1.]),('tanh+q3',lambda q,a:np.tanh(a[0]*q)+a[1]*q**3,[2.,.1,1.])]:
  def res(a):
   M=np.c_[np.ones(len(u)),u,fn(u,a)];return (M@L(M,v)-v).ravel()
  r=min([least_squares(res,[a,b,c],bounds=([.15,-10,.15],[8,10,8]),max_nfev=150) for a,b,c in [ini]+list(product([1.,2.5,4.],[-.25,.25],[1.,2.5,4.]))],key=lambda r:r.cost)
  print(k,nm,np.round(r.x,5),np.std(r.fun))
U,S,V=np.linalg.svd(z,full_matrices=False); R=np.c_[y,(U[:,:2]*S[:2])]; print('noise',np.std(z-(U[:,:2]*S[:2])@V[:2]));
for nm,q in coords(x).items():
 def gr(q):
  Q=phi(q,2);return np.c_[Q,np.sin(1.7*q),np.cos(1.7*q),np.tanh(1.7*q),np.arctan(2.3*q),np.cos(1.28*q)]
 for reg in [.01,.1,1.]:
  M=gr(q); es=[]
  for J in [np.arange(56,72),np.arange(16),np.arange(28,44),np.argsort(np.linalg.norm(x,axis=1))[-16:]]:
   I=np.setdiff1d(np.arange(len(x)),J); X=M/np.maximum(M[I].std(0),.05); X[:,0]=1; Q=X[I]; co=np.linalg.solve(Q.T@Q+reg*np.eye(Q.shape[1]),Q.T@R[I]); pred=X[J]@co; es.append([np.mean(abs(pred[:,0]-y[J])),np.mean(abs(pred[:,1:]@V[:2]-z[J]))]);
  print(nm,reg,np.asarray(es).round(5).T)
