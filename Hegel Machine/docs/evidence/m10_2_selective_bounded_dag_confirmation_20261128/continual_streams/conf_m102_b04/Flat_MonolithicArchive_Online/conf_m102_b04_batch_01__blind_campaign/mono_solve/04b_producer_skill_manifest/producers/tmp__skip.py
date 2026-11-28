from graph import *
from scipy.optimize import least_squares
W={k:Z[k] for k in ('r','a','f','rb','ab','fb')}; W.update({k+'sb':af(soft(W[k],.22))@B for k in ('r','a','f')})
K={}
for n,w in W.items():
 v=soft(w,.35); p=basis(v,3); q=basis(w,3); d=(w*w).sum(1); K[n+'poly']=p@p.T; g=q/(1+.12*d[:,None]); K[n+'rad']=g@g.T
 g=p/(1+.1*(v[:,0]+v[:,1])[:,None]**2); K[n+'sum']=g@g.T
 for k in ['v1','v2']:
  for b0,z in [('pre',Z[n[0]]),('post',w),('posts',soft(w,.22))]:
   c=panel(z,k); F=np.c_[af(c),c*c]; K[n+k+b0]=F@F.T
 d=((w[:,None]-w[None,:])**2).sum(2)
 for h in [1,2,4]:K[n+'rbf'+str(h)]=np.exp(-d/h**2)
 v=soft(w,.22); p=basis(v,3); g=p/(1+.22*(v[:,1]+v[:,2])[:,None]**2); K[n+'tsum']=g@g.T
 for a,g in [('traw',np.tanh(w)),('tsm',np.tanh(.5*w))]:
  F=basis(g,4); K[n+a]=F@F.T

def evalk(K,pen,sp=splits):
 E=[]
 for i,j in sp:
  m=Y[i].mean(0); P=K[np.ix_(i,i)]+pen*np.eye(len(i)); yp=K[np.ix_(j,i)]@np.linalg.solve(P,Y[i]-m)+m; e=np.abs(yp-Y[j]); E.append([e[:,0].mean(),e[:,1:].mean()]);
 E=np.array(E); return np.r_[E[0],E[1:].mean(0),E[1:].max(0)]
base={k:v for k,v in K.items() if k.endswith(('poly','rad','sum','traw','tsm','rbf1','rbf2','rbf4'))}; R=[]
for k,v in base.items():
 n=k.split('poly')[0] if k.endswith('poly') else next(z for z in sorted(W,key=len,reverse=True) if k.startswith(z));
 extra={k2:v2 for k2,v2 in K.items() if k2.startswith(n+'v')}; extra.update({n+'both'+d:K[n+'v1'+d]+K[n+'v2'+d] for d in ('pre','post','posts')});
 for l,u0 in [('',0.),*extra.items()]:
  for w in ([0.] if not l else [.03,.25,2.]):
   for p in [.0003,.003,.025,.12]:R.append((evalk(v+w*u0,p),(k,l,w,p)))
for i,j in [(0,2),(2,4),(4,2),(1,3),(3,5),(5,3)]:
 print('KERNEL',i,flush=True)
 for a,k in sorted(R,key=lambda z:z[0][i]+.35*z[0][j])[:10]: print(k,np.round(a,5))
_,_,V=np.linalg.svd(T[:52],full_matrices=False)
print('time levels',np.round(V[:3],4)); print('projections',np.std(T@V.T,0));print('mean',T.mean(0));print('corrs PC2',np.corrcoef(np.c_[T@V[:2].T, y, x].T)[1]);
# Compare whether rational decline or offset exponential explains the time dependence.
for f in (lambda t,p:1-p[0]*np.exp(-p[1]*t),lambda t,p:1-p[0]/(1+p[1]*t)**p[2],lambda t,p:1-p[0]*np.exp(-p[1]*t)-p[2]*np.exp(-p[3]*t)):
 q=V[0]/V[0,-1]; r=least_squares(lambda p:f(t,p)/f(t[-1],p)-q,[.95,.8,.5,2]);print('time error',r.x,np.round(r.fun,6))
