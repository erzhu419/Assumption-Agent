from operators import *
from scipy.optimize import minimize_scalar
p=s['panel_b']; u,v=p['u'],p['v']; tr=s['train']; x=coords(tr['x']); ti=s['eval_times']; y,T=tr['y'],tr['traj']; ids=np.arange(len(x)); folds=[ids[52:],ids[56:],np.argsort(np.linalg.norm(x,axis=1))[-16:],*np.array_split(np.random.default_rng(801).permutation(ids),4)]
F={'sin':(lambda x,k:np.sin(k*x)/k,2.8), 'tanh':(lambda x,k:np.tanh(k*x)/k,1.4),'rat':(lambda x,k:x/(1+k*x*x),.74),'sqrt':(lambda x,k:x/np.sqrt(1+k*x*x),1.), 'atan':(lambda x,k:np.arctan(k*x)/k,1.5),'erf':(lambda x,k:__import__('scipy').special.erf(k*x)*np.sqrt(np.pi)/2/k,1.2),'exp':(lambda x,k:x*np.exp(-k*x*x),.7)}

def learn(idx,f,k0,variant):
 U,V=u[idx],v[idx]
 def design(x,q):
  k=np.exp(q[0]); a=np.exp(q[1:]);
  if variant=='pure':return aff(f(x,k))
  if variant=='tied':return aff(x-a*f(x,k))
  if variant=='add':return aff(x+a*f(x,k))
  return np.c_[aff(x),f(x,k)]
 def r(q):
  A=design(U,q);return (A@fit(A,V)-V).ravel()
 q0=np.log([k0]+([1.06] if variant=='tied' else ([.5] if variant=='add' else [])))
 q=least_squares(r,q0,bounds=(-4,2),max_nfev=150).x
 c=fit(design(U,q),V)
 return np.exp(q),lambda x:design(x,q)@c

def score(z):
 E=[]; yD=np.c_[aff(z),z[:,0]*z[:,2],z[:,1]*z[:,3]]; A=aff(z)
 for j in folds:
  i=np.setdiff1d(ids,j)
  def basis(logr): return -np.expm1(-np.exp(logr)[:,None]*ti)
  def residual(logr):
   q=basis(logr); return (T[i]-T[i]@np.linalg.pinv(q)@q).ravel()
  q=basis(least_squares(residual,np.log([.4,1.8]),bounds=(-4,3)).x)
  E.append([np.abs(yD[j]@fit(yD[i],y[i])-y[j]).mean(),np.abs(A[j]@fit(A[i],T[i]@np.linalg.pinv(q))@q-T[j]).mean()])
 return np.array(E)
results=[]
for name,(f,k0) in F.items():
 for variant in ('pure','tied','add','free'):
  q,P=learn(np.arange(len(u)),f,k0,variant); r=P(u)-v; ep=[]
  for i in range(len(u)):
   _,Pi=learn(np.delete(np.arange(len(u)),i),f,k0,variant);ep.append(Pi(u[i:i+1])[0]-v[i]);
  E=score(P(x)); results.append((name,variant,q,np.sqrt(np.mean(r*r)),np.sqrt(np.mean(np.square(ep))),E[0],E[1],E[2],E[3:].max(0)))
np.set_printoptions(precision=6,suppress=True)
for row in sorted(results,key=lambda r:max(np.r_[r[5:][0],r[7],r[8]])):print(*row)
