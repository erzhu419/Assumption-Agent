import sys; sys.path.insert(0,'/holdout_task')
import batch_solver as m
from laws12 import *
from scipy.optimize import minimize_scalar
u,v=s['panel_a']['u'],s['panel_a']['v']; _,_,k=calfit('aff'); u,v=u[k],v[k]
for j in range(4):
 w=u[:,j]; q=u[:,(j+2)%4]; y=v[:,j]
 bases={'basic':(w,w**3),'cross':(w,w**3,w*q),'odd':(w,w**3,w**5),'mix':(w,w**3,w*q,q),'offset':(w,w**3,q**2),'all':(u,w**3,w**5,mon(u,2))}
 for nm,xx in bases.items():
  def f(p):
   G=aff(*xx,np.sin(p[0]*q+p[1])); return G@ls(G,y)-y
  p=least_squares(f,[2.4,0],bounds=([1.5,-.5],[3.5,.5]));print(j,nm,np.round(p.x,6),np.sqrt(np.mean(p.fun**2)))
print('bases CV')
t=s['train']; g=m._features(s,t['x']); c,d=g[:,5:9],g[:,9:13]; p=g[:,13:].reshape(-1,4,4); dot=np.trace(p,axis1=1,axis2=2); e=p[:,0,1]; tt=s['eval_times']; yt=t['y']; T=t['traj']; rng=np.random.default_rng(9873)
splits=[np.arange(72)]+[rng.permutation(72) for _ in range(29)]
for name,A,B in [('full',g,g),('sum',aff(c.sum(1),d.sum(1),dot),aff(c,d,dot,e)),('sumnobias',np.c_[c.sum(1),d.sum(1),dot],np.c_[c,d,dot,e]),('partial',aff(c,d,dot),aff(c,d,dot,e)),('cross',aff(c,d,dot),aff(c,d,dot,p.reshape(-1,16))),('mix',aff(c,d,dot,e),aff(c,d,dot,e)),('mix2',aff(c.sum(1)+d.sum(1),dot),aff(c,d,dot,e))]:
 vals=[]
 for order in splits:
  a,b=order[:52],order[52:]; time=least_squares(lambda r: (lambda f:f@ls(f,T[a].T)-T[a].T)(-np.expm1(-tt[:,None]*r)).ravel(),[.4,1.8],bounds=([.05,.9],[.85,4])).x; f=-np.expm1(-tt[:,None]*time); py=A[b]@ls(A[a],yt[a]); pt=B[b]@ls(B[a],ls(f,T[a].T).T)@f.T; vals.append([np.mean(abs(py-yt[b])),np.mean(abs(pt-T[b])),np.max(np.mean(abs(pt-T[b]),0))])
 print(name,'52/20',vals[0],'mean',np.mean(vals,0),'worst',np.max(vals,0))
