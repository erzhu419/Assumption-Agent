exec(open('/tmp/inspect_node01.py').read().split("print('times'")[0])
from scipy.optimize import minimize, minimize_scalar
G={'tanh':lambda x,k:np.tanh(k*x),'sin':lambda x,k:np.sin(k*x),'atan':lambda x,k:np.arctan(k*x),'soft':lambda x,k:x/np.sqrt(1+k*x*x),'cos':lambda x,k:np.cos(k*x),'exp':lambda x,k:np.exp(-k*x*x),'rat':lambda x,k:x*x/(1+k*x*x),'log':lambda x,k:np.log1p(k*x*x),'sqrt':lambda x,k:np.sqrt(1+k*x*x)}
chosen=[]
for target,names in [('v1',['tanh','sin','atan','soft']),('v2',['cos','exp','rat','log','sqrt'])]:
 v=c[target]; best=None
 for name in names:
  g=G[name]
  X=lambda x,k:F(x,x,g(x,k))
  obj=lambda k: np.mean((v-X(u,k)@fit(X(u,k),v))**2)
  for free in (False,True):
   k=minimize_scalar(obj,bounds=(.1,4),method='bounded').x
   if free:k=minimize(obj,np.repeat(k,4),bounds=[(.1,4)]*4,method='L-BFGS-B',options={'gtol':1e-12,'ftol':1e-14}).x
   Xp=X(u,k); err=loo(Xp,v).mean(); print(target,name,free,k,'rmse',obj(k)**.5,'loo',err,'coef norms',np.linalg.norm(fit(Xp,v),axis=1));
   if best is None or err<best[0]:best=(err,name,k,fit(Xp,v))
 _,name,k,b=best; chosen.append(F(q,q,G[name](q,k))@b);print('best',target,name,k,b.T)
a,b=chosen
p=lambda x:np.column_stack([x[:,i]*x[:,j] for i in range(4) for j in range(i,4)])
cv=lambda X,Y: np.mean(abs(X[52:]@fit(X[:52],Y[:52])-Y[52:]))
for name,X in [('q',F(q,q)),('q2',F(q,q,p(q))),('q2+a',F(q,q,p(q),a)),('q2+b',F(q,q,p(q),b)),('q2+a+b',F(q,q,p(q),a,b)),('q2+a+b+a*b',F(q,q,p(q),a,b,a*b)),('q2+a+b+a^2',F(q,q,p(q),a,b,a*a)),('q2+a+b+q*a',F(q,q,p(q),a,b,q*a)),('q2+a+b+q*b',F(q,q,p(q),a,b,q*b)),('q2+a+b+q*a+q*b',F(q,q,p(q),a,b,q*a,q*b)),('a+b',F(q,a,b)),('a+b+p(a)',F(q,a,b,p(a))),('a+b+p(b)',F(q,a,b,p(b))),('a+b+a*b',F(q,a,b,a*b)),('allquad',F(q,q,a,b,p(q),p(a),p(b)))]:
 print(name,'y',cv(X,y),'loo',loo(X,y),'traj',cv(X,z),'resid',np.std(y-X@fit(X,y)))
