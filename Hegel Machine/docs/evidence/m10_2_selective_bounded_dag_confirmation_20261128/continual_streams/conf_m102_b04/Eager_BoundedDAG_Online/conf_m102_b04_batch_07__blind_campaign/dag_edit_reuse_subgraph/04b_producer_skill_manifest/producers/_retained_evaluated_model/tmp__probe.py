exec(open('/tmp/explore.py').read().split('for label,u in')[0])
from scipy.optimize import least_squares, minimize_scalar
from scipy.spatial.distance import cdist
N=len(x); split=[(np.arange(52),np.arange(52,N))]+[(np.setdiff1d(np.arange(N),va),va) for j in range(3) for va in np.array_split(np.random.default_rng(j).permutation(N),4)]
def show(K,l):
 E=[]
 for i,j in split:
  P=K[np.ix_(j,i)]@np.linalg.solve(K[np.ix_(i,i)]+l*np.eye(len(i)),Y[i]); e=np.abs(P-Y[j]); E.append([e[:,0].mean(),e[:,1:].mean()]);
 E=np.array(E); return np.round(np.r_[E[0],E[1:].mean(0),E[1:].max(0)],5)
for k in ['v1','v2']:
 v=c[k]; u=c['u']; print(k,'range',v.min(),v.max());
 for d in [1,2]:
  F=basis(u,d); H=F@np.linalg.pinv(F); lv=(1-np.diag(H))[:,None];
  for fam,inv,direct in [('sinh',lambda v,p:np.arcsinh(v*p)/p,lambda z,p:np.sinh(z*p)/p),('asin',lambda v,p:np.arcsin(v*p)/p,lambda z,p:np.sin(z*p)/p),('atanh',lambda v,p:np.arctanh(v*p)/p,lambda z,p:np.tanh(z*p)/p),('soft',lambda v,p:v/np.sqrt(1-(p*v)**2),lambda z,p:z/np.sqrt(1+(p*z)**2))]:
   def err(p):
    w=inv(v,p); q=w-(w-H@w)/lv; return np.std(direct(q,p)-v)
   r=minimize_scalar(err,bounds=(1e-4,(3 if fam=='sinh' else .999/np.max(np.abs(v)))),method='bounded');print(d,fam,round(r.x,5),round(r.fun,6))
F=basis(c['u'],2); f2=fit(F,c['v2']); print('v2 coeffs\n',np.round(f2,3)); print('orthogonality B',np.round(fb[1:].T@fb[1:],3))
for name,u in [('raw',x),('cal',xa),('b',basis(x,1)@fb),('ab',basis(xa,1)@fb),('c',basis(x,1)@fc),('ac',basis(xa,1)@fc),('v2',basis(x,2)@f2),('av2',basis(xa,2)@f2)]:
 items=[]
 for k in [.0,.09,.3,.7]:
  U=sq(u,k); g=U@U.T/4; D=cdist(U,U,'sqeuclidean');
  for typ,K in [('p3',(1+g)**3),('p4',(1+g)**4),('mix',(1+g)**2+.2*g**3+.04*g**4+.008*g**5),('rb1',np.exp(-D/4)),('rb2',np.exp(-D/9)),('rb3',np.exp(-D/16))]:
   if name in ['v2','av2']:K=K+(1+x@x.T/4)**2
   for l in [1e-6,1e-4,.001,.01]:
    q=show(K,l);items.append((q[2],name,k,typ,l,q))
 print('kernel',sorted(items,key=lambda r:r[0])[:3])
