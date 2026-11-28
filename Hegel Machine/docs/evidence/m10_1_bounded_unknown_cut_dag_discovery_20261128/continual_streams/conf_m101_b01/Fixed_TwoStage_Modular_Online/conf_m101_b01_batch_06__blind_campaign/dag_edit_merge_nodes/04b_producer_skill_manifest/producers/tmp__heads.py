import numpy as np,pickle
from sklearn.model_selection import KFold
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import PolynomialFeatures
from scipy.linalg import solve
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; r=dict(np.load('/tmp/reg.npz')); x,u,b,c,d=[r[k] for k in ('x','u','b','c1','c2')]; z=s['train']['traj']; y=s['train']['y']; Y=np.c_[y,r['Y'][:,1]]; folds=[(np.arange(52),np.arange(52,72))]+list(KFold(4,shuffle=True,random_state=0).split(x));
p=s['panel_b']; U,V=p['u'],p['v']; a=np.linalg.lstsq(np.c_[np.ones(len(U)),U],V,rcond=None)[0];
for label,fun in [('tanh',np.tanh),('sin',np.sin),('soft',lambda x:x/(1+x*x)),('linear',lambda x:x)]:
 def fit(th):
  vv=fun(U@th[:16].reshape(4,4)+th[16:20]); return (vv@th[20:36].reshape(4,4)+th[36:]-V).ravel()
 t=np.r_[np.eye(4).ravel(),np.zeros(4),a[1:].ravel(),a[0]]; o=least_squares(fit,t,max_nfev=160); print('B mixed before',label,np.sqrt(np.mean(o.fun**2)),o.x[:20].round(3))
for f in (lambda x:np.tanh(x),lambda x:np.sin(x),lambda x:x/(1+x*x),lambda x:np.arctan(x)):
 def F(t,w):return np.c_[np.ones(len(w)),w,f(w*t[:4]),np.sin(w*t[4:])]
 def err(t):
  m=F(t,U); h=np.linalg.pinv(m);return ((m@(h@V)-V)/np.maximum(.02,1-np.sum(m*h.T,axis=1))[:,None]).ravel()
 o=least_squares(err,np.ones(8),bounds=(.1,4),max_nfev=100); print('B axis curve',np.sqrt(np.mean(o.fun**2)),o.x.round(3))
print('Y quantiles',np.quantile(Y,[0,.1,.25,.5,.75,.9,1],axis=0).round(3)); print('train rows u, y, amp, z first');print(np.c_[u,Y,z[:,0]].round(3))
def ev(K):
 out=[]
 for alpha in (1e-5,.0005,.005,.03,.15):
  ee=[]
  for a,h in folds:
   m=Y[a].mean(0); P=K[np.ix_(h,a)]@solve(K[np.ix_(a,a)]+alpha*np.eye(len(a)),Y[a]-m,assume_a='pos')+m;ee.append(np.mean(abs(P-Y[h]),axis=0))
  ee=np.array(ee);out.append((ee.max(0),ee[0],alpha))
 return out
results=[]
for name,w in [('u',u),('x',x),('b',b),('c1',c),('c2',d),('all',np.c_[u,b,c,d]),('bc',np.c_[b,c,d]),('ub',np.c_[u,b])]:
 w=(w-w.mean(0))/w.std(0); D=euclidean_distances(w,squared=True)/w.shape[1]; G=w@w.T/w.shape[1];
 for typ,M in [(f'rbf{g}',np.exp(-g*D)) for g in (.1,.3,.6,1,2,4)]+[(f'matern{g}',(1+g*np.sqrt(D))*np.exp(-g*np.sqrt(D))) for g in (.3,.7,1.5,3)]+[(f'poly{k}',(G+1)**k) for k in (2,3,4,5,6)]:
  for a,h,l in ev(M):results.append((name,typ,l,*a,*h))
for j in range(3,7):print('kernel best',j,sorted(results,key=lambda t:t[j])[:4]);
