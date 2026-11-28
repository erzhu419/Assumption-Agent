exec(open('/tmp/probe05.py').read().split("a=s['panel_a']")[0])
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import minimize
from itertools import combinations
from scipy.spatial.distance import cdist

def calibrate(s,X):
 a=s['panel_a'];u,v=a['u'],a['v']; co=[]
 def basis(x,j):
  r=x[:,j]; h=x[:,(j+1)%4];return np.c_[aff(x),r**3,h*r]
 for j in range(4):
  d=basis(u,j); k=np.linalg.norm(v-u,axis=1)<5; co.append(fit(d[k],v[k,j]))
 def transform(x):return np.column_stack([basis(x,j)@co[j] for j in range(4)])
 print('A residual',np.std((transform(u)-v)[k]));
 return np.array([least_squares(lambda r:transform(r[None])[0]-q,q,max_nfev=30).x for q in X])
x,y,z=[s['train'][k] for k in ('x','y','traj')];n=len(x);X=calibrate(s,np.r_[x,s['test']['x']]); Y=np.c_[y,z]; print('latent ranges',X[:n].min(0),X[:n].max(0));print('y by q', np.c_[X[:n],y,z[:,0],z[:,-1]][:12]);
b=s['panel_b'];u=b['u'];v=b['v']; bl=[]
for name,D in [('cubic',np.c_[aff(u),u**3]),('onlyc',aff(u**3)),('c+pair',np.c_[aff(u),u**3,*[u[:,i]*u[:,j] for i,j in combinations(range(4),2)]]),('c+trip',np.c_[aff(u),u**3,*[np.prod(u[:,j],axis=1) for j in combinations(range(4),3)]])]:
 B=fit(D,v);print('B',name,np.std(D@B-v),B.round(3).T)
 if name=='cubic':bl=B

a,beta=1.8275,.4277; t=s['eval_times'];T=np.array([-np.expm1(-a*t),-np.expm1(-beta*t)]).T; A=fit(T,z.T).T;Y0=np.c_[y,A];
c=s['panel_c']; cfun=lambda u:aff(np.arctan(2.1143*u)); dfun=lambda u:np.c_[aff(u),u*u/(1+.1653*u*u)]; C=cfun(X)@fit(cfun(c['u']),c['v1']);D=dfun(X)@fit(dfun(c['u']),c['v2']);BX=np.c_[aff(X),X**3]@bl

def summary(P,name):
 for j in range(3):
  b=fit(P[:52],Y0[:52,j]); print(name,j,np.std(P[:52]@b-Y0[:52,j]),np.mean(abs(P[52:n]@b-Y0[52:,j])),b.round(3))
summary(poly(X,2),'poly2');summary(poly(X,3),'poly3'); summary(aff(np.c_[C,D]),'C1C2')

def cv(k,pen, ids,js):
 f=cho_factor(k[np.ix_(ids,ids)]+pen*np.eye(len(ids))); p=Y[ids].mean(0)+k[np.ix_(js,ids)]@cho_solve(f,Y[ids]-Y[ids].mean(0));e=abs(p-Y[js]).mean(0);return e[0],e[1:].mean()

ij=[(np.arange(52),np.arange(52,72))]; rng=np.random.default_rng(42)
for _ in range(4): jj=rng.permutation(n);ij.append((jj[20:],jj[:20]))
res=[]
for zn,w in [('lat',X),('C',C),('D',D),('B',BX),('atan',np.arctan(2.1143*X)),('cubic',X**3),('tanh',np.tanh(X)),('lat+C',np.c_[X,C]),('lat+D',np.c_[X,D]),('lat+B',np.c_[X,BX])]:
 w=w[:n];w=(w-w.mean(0))/w.std(0);sq=(w[:,None]-w[None,:])**2; d=sq.mean(2); g=w@w.T/w.shape[1]
 def add(k,desc):
  for p in [.00003,.0003,.003,.03,.15]:
   e=np.array([cv(k,p,i,j) for i,j in ij]);res.append((e[0,0],e[0,1],*e[1:].max(0),zn,desc,p))
 for gam in [.02,.08,.2,.5,1,2]:
  r=np.exp(-gam*d);ad=np.exp(-gam*sq).mean(2);m=np.sqrt(3*gam*d); add(r,('rbf',gam));add(.5*r+.5*ad,('add',gam));add((1+m)*np.exp(-m),('m32',gam));add(r+(1+g)**3*.05,('R&P',gam))
 for dd in [2,3,4,5,6]:add((1+g)**dd,('poly',dd))
for i in [0,1,2,3]: print('BEST',i,*sorted(res,key=lambda x:x[i])[:8],sep='\n')
