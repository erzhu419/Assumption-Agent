exec(open('/tmp/discover.py').read().split('maps=[]')[0])
from scipy.optimize import minimize_scalar

def panel1(u,v):
 def phi(x,p):
  h=np.abs(x)*np.exp(p[:4]); k=p[4]; a=np.log1p(h)
  return np.sign(x)*a*(1+k*a) if len(p)==5 else np.sign(x)*np.expm1(k*a)/k
 for kind in ('log','logplus','power'):
  def G(x,p):
   h=np.abs(x)*np.exp(p[:4]); a=np.log1p(h)
   if kind=='logplus': a=a*(1+p[4]*a)
   if kind=='power': a=np.expm1(p[4]*a)/p[4]
   return aug(np.c_[x,np.sign(x)*a])
  def fun(p):
   g=G(u,p);return (g@fit(g,v)-v).ravel()
  extra=[] if kind=='log' else [.1]; r=least_squares(fun,np.r_[np.full(4,.4),extra],bounds=(np.r_[np.full(4,-3),[-3]*len(extra)],np.r_[np.full(4,3),[3]*len(extra)]),ftol=1e-12,gtol=1e-12)
  g=G(u,r.x);print('C1 shape',kind,loo(g,v),r.x,'coefs',fit(g,v).round(3),flush=True)
  if kind=='logplus':
   p=r.x.copy();bb=fit(g,v)
   def out(x,p=p,bb=bb):
    h=np.log1p(np.abs(x)*np.exp(p[:4]));return aug(np.c_[x,np.sign(x)*h*(1+p[4]*h)])@bb
 return out

def panel2(u,w):
 sq=u*u; best=None
 for name,fn in {'rational':lambda x,a:x*x/(1+a*x*x),'tanh':lambda x,a:np.tanh(a*x*x),'cos':lambda x,a:1-np.cos(a*x),'sqrt':lambda x,a:np.sqrt(1+a*x*x)-1,'log':lambda x,a:np.log1p(a*x*x),'exp':lambda x,a:-np.expm1(-a*x*x),'quartic':lambda x,a:x*x*(1+a*x*x),'cube':lambda x,a:x*x*(1+a*x)}.items():
  def G(x,a,fn=fn):return aug(np.c_[x,fn(x,a)])
  r=minimize_scalar(lambda a:loo(G(u,a),w)[0],bounds=(.0001,1.8) if name not in ('quartic','cube') else (-.4,.4),method='bounded');g=G(u,r.x);e=loo(g,w);print('C2 shape',name,r.x,e,flush=True)
  if best is None or e[1]<best[0]:best=e[1],(lambda x,G=G,a=r.x,b=fit(g,w):G(x,a)@b)
 return best[1]
C=panel1(u,v);D=panel2(u,w)
x=s['train']['x'];Y=np.c_[s['train']['y'],s['train']['traj']];t=s['eval_times'];z=Y[:,1:]; print('temporal model')
for f in [lambda t,a:np.log1p(a*t),lambda t,a:-np.expm1(-a*t),lambda t,a:t/(1+a*t),lambda t,a:np.arctan(a*t)]:
 def gram(p):return np.c_[f(t,p[0]),f(t,p[1])]
 r=least_squares(lambda p:(gram(p)@fit(gram(p),z.T)-z.T).ravel(),[.5,3],bounds=(.001,50));print(r.x,np.sqrt(np.mean(r.fun**2)))

rs=np.random.RandomState(228); folds=[(np.arange(52),np.arange(52,72))]+[(np.setdiff1d(np.arange(72),j),j) for j in np.array_split(rs.permutation(72),5)]; radial=np.argsort(np.linalg.norm(x,axis=1))[-15:];folds.append((np.setdiff1d(np.arange(72),radial),radial)); results=[]
for name,F in {'B':B(x),'AB':A(B(x)),'BA':B(A(x)),'raw':x}.items():
 c=C(F);d=D(F); pairs=[(i,j) for i in range(4) for j in range(i+1,4)]
 parts={'f':F,'c':c,'d':d,'m':(F[:,:,None]*c[:,None,:]).reshape(len(F),-1),'f2':F*F,'c2':c*c,'cc':np.column_stack([c[:,i]*c[:,j] for i,j in pairs]),'ff':np.column_stack([F[:,i]*F[:,j] for i,j in pairs])}
 for mask in range(1,1<<len(parts)):
  keys=[k for i,k in enumerate(parts) if mask>>i&1];G=np.column_stack([parts[k] for k in keys]);
  if G.shape[1]>44:continue
  for lam in (.00003,.0003,.003,.03):
   errs=[]
   for tr,va in folds:
    H=aug(G[tr]);pen=np.repeat(lam,H.shape[1]);pen[0]=0; e=np.abs(aug(G[va])@np.linalg.solve(H.T@H+np.diag(pen),H.T@Y[tr])-Y[va]);errs.append([e[:,0].mean(),e[:,1:].mean(),e[:,-1].mean()])
   e=np.array(errs); obj=(e[1:6].mean(0)[0]+e[1:6].mean(0)[1]+.5*(e[-1,0]+e[-1,1]));results.append((obj,name,','.join(keys),lam,e[0],e[1:6].mean(0),e[1:6].max(0),e[-1]))
for row in sorted(results,key=lambda r:r[0])[:18]: print('cv',row)
