exec(open('/tmp/probe.py').read().split('np.set_printoptions')[0])
from scipy.optimize import least_squares
q=u.T; z=p['v2']; N=len(u)
P=np.column_stack([np.ones(N),*q,*[q[i]*q[j] for i in range(4) for j in range(i,4)]])
sets={'additive':np.c_[np.ones(N),q.T,(q*q).T], 'squares':np.c_[np.ones(N),(q*q).T], 'quad':P}
for name,A in sets.items():
 c=np.linalg.lstsq(A,z,rcond=None)[0]; H=np.sum(np.linalg.pinv(A).T*A,axis=1);print(name,'fit',np.std(z-A@c,axis=0),'loo',np.mean(abs((z-A@c)/(1-H[:,None])),axis=0))
 if name!='quad':print(np.round(c,3))
for typ,A in [('additive',sets['additive']),('quad',P)]:
 for f in (np.sin,np.tanh):
  for j in range(4):
   v=z[:,j]; best=None
   def res(w):return w[-2]*f(A@w[:-2])+w[-1]-v
   rng=np.random.default_rng(13)
   for k in range(24):
    ini=np.r_[rng.normal(0,.3,A.shape[1]),1.5,0.];r=least_squares(res,ini,max_nfev=100); e=np.std(r.fun)
    if best is None or e<best[0]:best=e,r.x
   print('link',typ,f.__name__,j,round(best[0],5),np.round(best[1],2))
# Search a shared trig/saturation grammar by forward selection with re-projection.
D={'1':np.ones(N)}
for i in range(4):
 for j in range(i,4):D[f'p{i}{j}']=q[i]*q[j]
 D[f'x{i}']=q[i]
for ix in itertools.combinations_with_replacement(range(4),3):D['p'+''.join(map(str,ix))]=np.prod(q[list(ix)],axis=0)
for i in range(4):
 for j in range(i,4):
  items=[(str(i),q[i]),(f'{i}^2',q[i]**2)] if i==j else [(f'{i}+{j}',q[i]+q[j]),(f'{i}-{j}',q[i]-q[j]),(f'{i}*{j}',q[i]*q[j]),(f'{i}^2-{j}^2',q[i]**2-q[j]**2),(f'{i}^2+{j}^2',q[i]**2+q[j]**2)]
  for k,w in items:
   for g in (1,1.5,2,2.5,3,4):
    for f in (np.sin,np.cos,np.tanh):D[f'{f.__name__}{g}({k})']=f(g*w)
  if i!=j:
   for a,b in ((i,j),(j,i)):
    for g in (.5,1,2,3):
     for h in (1,2):D[f'{a}^{h}/(1+{g}*{b}^2)']=q[a]**h/(1+g*q[b]**2)
names=list(D);F=np.array(list(D.values())).T
for jj in range(5):
 Y=z if jj==4 else z[:,jj:jj+1]; Q=np.ones((N,1))/np.sqrt(N); sel=[0]
 for t in range(12):
  G=F-Q@(Q.T@F); d=np.linalg.norm(G,axis=0);V=G/np.maximum(d,1e-9);i=np.argmax(np.linalg.norm(V.T@Y,axis=1));sel.append(i);Q=np.c_[Q,V[:,i]]
  if t in (3,6,11):print('greedy',jj,t,np.round(np.std(Y-Q@(Q.T@Y),axis=0),5),[names[k] for k in sel])
