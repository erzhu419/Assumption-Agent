from operators import *
from scipy.spatial.distance import cdist
from scipy.linalg import solve
D=S['train']; x=D['x']; y=D['y']; z=D['traj']; t=S['eval_times']; n=len(x)
F=np.c_[affine(x),powers(x,2)]; G=powers(x,3)
rng=np.random.default_rng(451); splits=[(np.arange(56),np.arange(56,72))]+[(p[:56],p[56:]) for p in [rng.permutation(72) for _ in range(8)]]
def ridge(x, v, kind):
 w=x@v[:4]; a=np.exp(v[4])
 if kind=='p':return np.c_[w**3,w**4,w**5]
 if kind=='sin':return np.sin(a*w+v[5])[:,None]
 if kind=='tanh':return np.tanh(a*w+v[5])[:,None]
 if kind=='rat':return (1/(1+(a*w+v[5])**2))[:,None]
 if kind=='quot':return np.c_[np.ones(len(x)),x]/(1+(a*w[:,None]+v[5])**2)
 if kind=='norm':return (1/(1+a*np.sum((x-v[:4])**2,axis=1)))[:,None]
 if kind=='soft':return ((a*w+v[5])/(1+abs(a*w+v[5])))[:,None]
def fit(x,F,Y,kind,seeds):
 Q=np.linalg.qr(F)[0]; r0=Y-Q@(Q.T@Y)
 def fun(v):
  H=ridge(x,v,kind); H=H-Q@(Q.T@H)
  return (r0-H@ls(H,r0)).ravel()
 lo=np.array([-2.]*4+[-2.,-3.]); hi=-lo
 fs=[least_squares(fun,p,bounds=(lo,hi),max_nfev=140,ftol=1e-7) for p in seeds]
 f=min(fs,key=lambda p:p.cost); H=np.c_[F,ridge(x,f.x,kind)]; return f.x,ls(H,Y),np.sqrt(np.mean(fun(f.x)**2))
results={}; params={}
for tr,te in splits:
 m=modes(t,z[tr]); A=ls(m,z[tr].T).T
 Q=np.linalg.qr(F[tr])[0]; RR=y[tr]-Q@(Q.T@y[tr]); C=G[tr]-Q@(Q.T@G[tr]); cb=ls(C,RR)
 ten=np.zeros((4,4,4)); from itertools import permutations
 for idx,val in zip(combinations_with_replacement(range(4),3),cb):
  perms=set(permutations(idx));
  for j in perms:ten[j]=val/len(perms)
 u=np.linalg.svd(ten.reshape(4,-1))[0].T
 for j,Y in enumerate((y[tr,None],A)):
  for kind in ('p','sin','tanh','rat','quot','norm','soft'):
   seeds=[np.r_[v,l,h] for v in u[:3] for l,h in ((0.,0.),(.25,.6),(-.4,-.6))]
   p,b,e=fit(x[tr],F[tr],Y,kind,seeds); pr=np.c_[F[te],ridge(x[te],p,kind)]@b
   target=y[te,None] if j==0 else z[te]; pr=pr if j==0 else pr@m.T
   results.setdefault((j,kind),[]).append(np.mean(abs(pr-target))); params.setdefault((j,kind),(p,e))
 for label,l,w,no in [('short',1.1,.1,.001),('long',1.6,1,.001),('verylong',2.3,1,.0001)]:
  K=F@F.T+w*np.exp(-cdist(x,x,'sqeuclidean')/(2*l*l)); P=K[np.ix_(te,tr)]@solve(K[np.ix_(tr,tr)]+no*np.eye(len(tr)),np.c_[y[tr],A],assume_a='pos');
  results.setdefault((0,label),[]).append(np.mean(abs(P[:,0]-y[te]))); results.setdefault((1,label),[]).append(np.mean(abs(P[:,1:]@m.T-z[te])))
for k,v in results.items():print(k,'gate mean worst',np.round([v[0],np.mean(v),max(v)],6),'first fit',params.get(k))
