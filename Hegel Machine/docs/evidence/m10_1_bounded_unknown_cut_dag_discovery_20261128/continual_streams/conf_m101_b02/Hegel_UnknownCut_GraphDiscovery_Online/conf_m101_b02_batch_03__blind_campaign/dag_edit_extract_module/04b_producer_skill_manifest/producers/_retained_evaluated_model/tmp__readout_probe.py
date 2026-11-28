exec(open('/tmp/dag_probe.py').read().split("if '--setup'")[0])
from scipy.linalg import qr
from itertools import combinations
H=-np.expm1(-t[:,None]*np.array([.42,1.8])); A=T@np.linalg.pinv(H.T); Y=np.c_[y,A]; print('amplitude range and correlations',np.quantile(Y,[0,.5,1],axis=0),np.corrcoef(Y.T),flush=True)
folds=[(np.arange(52),np.arange(52,72))]; rng=np.random.default_rng(25); folds += [(np.setdiff1d(np.arange(72),v),v) for v in np.array_split(rng.permutation(72),4)]
feat={'lin':lambda x:x,'sin':np.sin,'cos':np.cos,'tanh':np.tanh,'soft':lambda x:x/(1+abs(x)),'sq':lambda x:x*x,'cub':lambda x:x**3,'abs':abs,'exp':lambda x:np.exp(-x*x),'rat':lambda x:x/(1+x*x)}
base={'z':z,'B':b,'F':f,'C':c,'D':d,'x':x};
def basemat(q, names):
 return np.c_[np.ones(n),*[feat[k](q) for k in names]]
def ridge(X,y,a,v,lam):
 scale=np.maximum(np.std(X[a],axis=0),.001);scale[0]=1;P=X/scale; W=P[a];return P[v]@np.linalg.solve(W.T@W+lam*np.eye(W.shape[1]),W.T@y[a])
def report(P):
 return np.array([abs(P[:,0]).mean(),abs(P[:,1:]@H.T).mean()])
rows=[]
for name,q in list(base.items())+[(a+k,np.c_[base[a],base[k]]) for a,k in combinations(base,2)]+[('zBCD',np.c_[z,b,c,d]),('zFCD',np.c_[z,f,c,d])]:
 for fs in [('lin','sin'),('lin','tanh'),('lin','sq'),('lin','sq','cub'),('lin','sin','cos'),('lin','tanh','cos'),('lin','soft','abs'),('lin','rat','exp'),('lin','sq','cub','sin','cos','tanh')]:
  X=basemat(q,fs)
  for lam in [.00001,.001,.02,.3]:
   E=np.array([report(ridge(X,Y,a,v,lam)-Y[v]) for a,v in folds]); rows.append((E[0],E[1:].max(axis=0),name,fs,lam))
for j in range(2): print('additive best',j,sorted(rows,key=lambda r:r[0][j])[:8],flush=True)
# A train-only search over single-index and additive projection mechanisms.
funcs={'sin':np.sin,'tanh':np.tanh,'cos':np.cos,'soft':lambda q:q/np.sqrt(1+q*q),'mix':lambda q:np.column_stack((np.sin(q[:,0]),np.tanh(q[:,1])))}
def fit(q,lin,a,v,yy,kind,m,seed):
 q=q/np.maximum(q[a].std(axis=0),.01); qq=np.c_[np.ones(len(q)),q]; X=L(lin); xa=X[a]; U=np.linalg.svd(xa,full_matrices=False)[0][:,:xa.shape[1]]; yo=yy[a]-U@(U.T@yy[a]); p=qq.shape[1]; rng=np.random.default_rng(seed); fun=funcs[kind]; best=None
 def residual(theta):
  N=fun(qq[a]@theta.reshape(p,m)); N-=U@(U.T@N); coef=np.linalg.lstsq(N,yo,rcond=1e-8)[0];return np.r_[N@coef-yo, .002*theta]
 for _ in range(6):
  opt=least_squares(residual,rng.normal(0,.45,p*m),max_nfev=95)
  loss=opt.fun@opt.fun
  if best is None or loss<best[0]:best=(loss,opt.x)
 N=fun(qq@best[1].reshape(p,m)); P=np.c_[X,N]; C=np.linalg.lstsq(P[a],yy[a],rcond=1e-8)[0];return P[v]@C,np.sqrt(best[0]/len(a))
results=[]
for name,q in base.items():
 for linear in ['z',name] if name!='z' else ['z']:
  for kind,m in [('sin',1),('tanh',1),('cos',1),('sin',2),('tanh',2),('mix',2),('soft',2),('sin',3)]:
   a,v=folds[0]; ps=[];tr=[];ev=[]
   for j in range(3):
    p,r=fit(q,base[linear],a,v,Y[:,j],kind,m,10+j); ps.append(p);tr.append(round(r,5));ev.append(round(np.mean(abs(p-Y[v,j])),5))
   e=report(np.column_stack(ps)-Y[v]);results.append((e,name,linear,kind,m,tr,ev))
for j in range(2): print('projection best',j,sorted(results,key=lambda r:r[0][j])[:12],flush=True)
for j in range(3): print('amplitude best',j,sorted(results,key=lambda r:r[-1][j])[:6],flush=True)
