from features import *
from scipy.linalg import solve, cholesky, solve_triangular
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import io,contextlib
with contextlib.redirect_stdout(io.StringIO()):
 ns={}; exec(open('/tmp/compose.py').read().split("print('temporal residual'")[0],ns); c,d=ns['c'],ns['d']
idx=np.arange(n)
def err(e):return np.round([e[0],np.median(e[1:]),np.max(e[1:])],6)
for deg in range(1,4):
 for name,W in [('x',x),('ad',d(coords['a'])),('abc',c(coords['ab']))]:
  H=pol(np.c_[W,h],deg); print('ORACLE',deg,name,err([np.mean(abs(H[j]@ls(H[i],y[i])-y[j])) for i,j in splits]))
# Compare smoothness and directional scaling with training-only restricted likelihood.
from sys import path
path.insert(0,'/holdout_task')
from batch_solver import _gp
U={'x':x,'ab':coords['ab'],'ad':d(coords['a']),'abc':c(coords['ab'])}

def model(i,j,y0,kind,coord='x',mean=2,amp=None):
 w=U[coord]; w=w/w[i].std(0); a=w[i]; b=w[j]; H=pol(x,mean); f,g=H[i],H[j]
 if amp is not None:
  hi,hj=amp; f=np.c_[f,hi,hi**2,hi[:,None]*x[i]];g=np.c_[g,hj,hj**2,hj[:,None]*x[j]]
 N=np.linalg.qr(f,mode='complete')[0][:,f.shape[1]:]; Y0=N.T@y0[i]; E=np.eye(len(i)); D=(a[:,None,:]-a[None,:,:])**2
 def make(D,p):
  z=D@np.exp(-2*p[:-1]);
  if kind in ('e','iso'): return np.exp(-z/2)
  if kind=='m':
   r=np.sqrt(5*z);return (1+r+r*r/3)*np.exp(-r)
  return (1+z/2)**-2
 def loss(p):
  k=make(D,p)+np.exp(p[-1])*E; K=N.T@k@N; L=cholesky(K,lower=True); v=solve_triangular(L,Y0,lower=True); return np.log(np.diag(L)).sum()+len(v)*.5*np.log(v@v/len(v))+.02*np.sum(p[:-1]**2)
 def pp(p):return np.r_[np.repeat(p[0],4),p[-1]] if kind=='iso' else p
 opt=minimize(lambda p:loss(pp(p)),np.r_[np.full(1 if kind=='iso' else 4,.6),-8.],bounds=[(-1.5,2.6)]*(1 if kind=='iso' else 4)+[(-15,-1)],method='L-BFGS-B',options={'maxiter':80}); p=pp(opt.x)
 k=make(D,p)+np.exp(p[-1])*E; K=make((b[:,None,:]-a[None,:,:])**2,p); L=cholesky(k,lower=True); v=solve_triangular(L,np.c_[f,y0[i]],lower=True); cf=ls(v[:,:-1],v[:,-1]); return g@cf+K@solve_triangular(L.T,v[:,-1]-v[:,:-1]@cf)
if __name__=='__main__':
 preds={}
 for kind in ['iso','e','m','r']:
  for coord in U:
   for mean in [1,2]:
    for joint in ([False,True] if coord=='x' else [False]):
     ps=[]
     for i,j in splits:
      amp=(h[i],_gp(x[i],x[j],h[i],2.4,.001,True)) if joint else None
      ps.append(model(i,j,y,kind,coord,mean,amp))
     k=(kind,coord,mean,joint);preds[k]=ps
     print(k,err([np.mean(abs(a-y[j])) for a,(i,j) in zip(ps,splits)]),flush=True)
 for name,keys in [('plain',[k for k in preds if not k[-1]]),('joint',[k for k in preds if k[-1]]),('raw',[k for k in preds if k[1]=='x' and not k[-1]])]:
  p=np.mean([preds[k] for k in keys],0);print('ensemble',name,err([np.mean(abs(a-y[j])) for a,(i,j) in zip(p,splits)]),flush=True)
 # Localize baseline failures to see whether a separate branch or extrapolation dominates.
 g=_gp(x[:52],x[52:],y[:52],1.5,.0001); q=np.argsort(abs(g-y[52:]))[-8:]; print('largest baseline residuals',np.round(np.c_[x[52:][q],h[52:][q],y[52:][q],g[q]],4))
