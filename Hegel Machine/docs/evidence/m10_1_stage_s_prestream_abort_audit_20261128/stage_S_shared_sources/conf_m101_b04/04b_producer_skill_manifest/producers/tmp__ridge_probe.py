from probe import *
from scipy.linalg import polar
from scipy.optimize import minimize_scalar
from itertools import combinations
p=s['panel_c']; u=p['u']; J=lambda u,a:np.sign(u)*np.log1p(a*abs(u));
q=minimize_scalar(lambda a:np.linalg.norm(p['v1']-np.c_[np.ones(len(u)),u,J(u,a)]@fit(np.c_[np.ones(len(u)),u,J(u,a)],p['v1'])),bounds=(.9,2.1),method='bounded'); L=fit(np.c_[np.ones(len(u)),u,J(u,q.x)],p['v1']); R=polar(L[-4:])[0]; print('C1 approximation',q.x,L[1:5]@R.T,L[-4:]@R.T)
print('C2 diagonal polys',fit(np.c_[np.ones(len(u)),u,u**2,u**3,u**4],p['v2']))
p=s['panel_a']; aa=[]
for i in range(4):
 X=poly(p['u'][:,i:i+1],1); Y=p['v'][:,i]; b=fit(X,Y); k=np.argsort(abs(Y-X@b))[:-3]; aa.append(fit(X[k],Y[k]))
a,g=np.array(aa).T
p=s['panel_b']; L=fit(poly(p['u'],1),p['v']); B=polar(L[1:])[0]
x=s['train']['x']; Z=np.c_[s['train']['y'],s['train']['traj']]; bases={'x':x,'A':x*g+a,'Ai':(x-a)/g}
for k,w in list(bases.items()):
 for j,M,b in [('B',L[1:],L[0]),('Bi',np.linalg.inv(L[1:]),-L[0]@np.linalg.inv(L[1:])),('C',R,0),('Ci',R.T,0)]:bases[k+j]=w@M+b
for k,w in list(bases.items()):
 if len(k)>2:
  bases[k+'R']=w@R
  bases[k+'Ri']=w@R.T
folds=[(np.arange(56),np.arange(56,72))]+[(np.setdiff1d(np.arange(72),v),v) for v in np.array_split(np.random.default_rng(42).permutation(72),4)]
def score(X,lam=0):
 err=[]
 for t,v in folds:
  U,d,V=np.linalg.svd(X[t],full_matrices=False); b=V.T@((d/(d*d+lam+1e-15))[:,None]*(U.T@Z[t])); err.append(np.mean(abs(X[v]@b-Z[v]),axis=0))
 err=np.array(err); return np.array([err[:,0].mean(),err[:,1:].mean(),err[:,0].max(),err[:,1:].max()])
rows=[]
for k,w in bases.items():
 W=np.c_[np.ones(len(w)),w]; W0=poly(x,2)
 for j,f in [('log',lambda w:J(w,1.5)),('sin',np.sin),('tanh',np.tanh),('soft',lambda w:w/np.sqrt(1+w*w)),('rational',lambda w:w/(1+w*w)),('sin.5',lambda w:np.sin(.5*w)),('tanh.5',lambda w:np.tanh(.5*w)),('abs',abs),('cube',lambda w:w**3),('square',lambda w:w*w)]:
  for k2,F in [('lin',W),('quad',W0),('diag2',np.c_[W,w*w])]:
   rows.append((score(np.c_[F,f(w)],.00001),k,j,k2))
 for d in (2,3,4,5,6):
  for lam in (0,.0001,.002,.02): rows.append((score(np.c_[W0,*[w**i for i in range(3,d+1)]],lam),k,'degree',d,lam))
for i in range(2):
 print('HEAD',i)
 for r in sorted(rows,key=lambda r:r[0][i])[:14]:print(r)
print('raw cubic',score(poly(x,3)))
print('train test range',np.quantile(np.linalg.norm(x,axis=1),[0,.25,.5,.75,1]),np.quantile(np.linalg.norm(s['test']['x'],axis=1),[0,.25,.5,.75,1]))
