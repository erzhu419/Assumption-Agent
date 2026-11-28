import pickle
import numpy as np
from itertools import combinations_with_replacement
from scipy.optimize import minimize_scalar
from scipy.linalg import cho_factor,cho_solve
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; Y=np.c_[s['train']['y'],s['train']['traj']]
def aff(z): return np.c_[np.ones(len(z)),z]
def mono(z,d):return np.column_stack([np.prod(z[:,j],axis=1) for j in combinations_with_replacement(range(z.shape[1]),d)])
def panel(key,out,feat,trim=False):
 d=s[key]; X=feat(d['u']); y=d[out]; i=np.arange(len(y))
 for _ in range(5 if trim else 1):
  w=np.linalg.lstsq(X[i],y[i],rcond=None)[0]; i=np.argsort(np.linalg.norm(X@w-y,axis=1))[:len(y)-3] if trim else i
 return lambda z:feat(z)@w
A=panel('panel_a','v',aff,True); B=panel('panel_b','v',aff); Ai=lambda z:(z-A(np.zeros((1,4))))@np.linalg.inv(A(np.eye(4))-A(np.zeros((1,4))))
u=s['panel_c']['u']; V=s['panel_c']['v1']; W=s['panel_c']['v2']
def score(f,Y):
 D=f(u); P=np.linalg.pinv(D); E=(Y-D@P@Y)/(1-np.sum(D*P.T,axis=1))[:,None]; return np.mean(E**2)**.5
res=[]
for n,g in [('tanh',np.tanh),('sin',np.sin),('rat',lambda z:z/(1+z*z)),('soft',lambda z:z/np.sqrt(1+z*z)),('atan',np.arctan),('erf',__import__('scipy').special.erf)]:
 for q in [False,True]:
  def ff(z,k):return np.c_[aff(z),g(k*z),z*z] if q else np.c_[aff(z),g(k*z)]
  r=minimize_scalar(lambda k:score(lambda z:ff(z,k),V),bounds=(.15,3.5),method='bounded');print('C1',n,q,r.x,r.fun,flush=True)
  res.append((r.fun,n,q,r.x,lambda z,g=g,k=r.x,q=q:np.c_[aff(z),g(k*z),z*z] if q else np.c_[aff(z),g(k*z)]))
C=panel('panel_c','v1',min(res,key=lambda a:a[0])[-1]); D=panel('panel_c','v2',lambda z:np.c_[aff(z),z*z]);print('C2 additive LOO',score(lambda z:np.c_[aff(z),z*z],W));print('C2 sqrt LOO',score(lambda z:aff(z+.45*z*z),W));
rng=np.random.default_rng(228); splits=[(np.arange(52),np.arange(52,72))]+[(p[:52],p[52:]) for p in (rng.permutation(72) for _ in range(8))]
def metric(pred,Y):return np.array([np.mean(abs(pred[:,0]-Y[:,0])),np.mean(abs(pred[:,1:]-Y[:,1:]))])
def cv(P,K,al):
 E=[]
 for a,b in splits:
  p=P[a]; f=cho_factor(K[np.ix_(a,a)]+al*np.eye(len(a))); aa=cho_solve(f,p); yy=cho_solve(f,Y[a]); c=np.linalg.lstsq(p.T@aa,p.T@yy,rcond=1e-9)[0]; pred=P[b]@c+K[np.ix_(b,a)]@(yy-aa@c); E.append(metric(pred,Y[b]))
 E=np.array(E);return np.array([E[0],E.mean(0),E.max(0)])
Z={'x':x,'A':A(x),'Ai':Ai(x),'B':B(x),'AB':B(A(x)),'AiB':B(Ai(x))}; X=dict(Z)
for n,z in list(Z.items()):
 for j,f in [('C',C),('D',D)]: X[n+j]=f(z)
for n,z in list(X.items()):
 if n.endswith(('C','D')):
  X[n+'B']=B(z)
  if n.endswith('C'):X[n+'D']=D(z)
  else:X[n+'C']=C(z)
Q=np.c_[aff(x),mono(x,2)]; K0=np.exp(-.23*np.sum((x[:,None]-x[None,:])**2,axis=2)); R=[]
for n,z in X.items():
 P=np.c_[aff(z),mono(z,2)]; z=(z-z.mean(0))/z.std(0); K=np.exp(-.13*np.sum((z[:,None]-z[None,:])**2,axis=2)); L=(z@z.T/4+1)**2
 for k,kern in [('K',K),('mix',K+.4*K0+.05*L),('poly',.01*K0+L)]:
  for p,pp in [('x',Q),('z',P),('lin',aff(x))]:
   for al in [.0002,.003,.025]: R.append((cv(pp,kern,al),n,k,p,al))
for j in range(2):
 for how in [0,1]:
  print('target',j,'ordering',how)
  for r,n,k,p,al in sorted(R,key=lambda v:v[0][how,j])[:7]:print(n,k,p,al,np.round(r,5).tolist())
