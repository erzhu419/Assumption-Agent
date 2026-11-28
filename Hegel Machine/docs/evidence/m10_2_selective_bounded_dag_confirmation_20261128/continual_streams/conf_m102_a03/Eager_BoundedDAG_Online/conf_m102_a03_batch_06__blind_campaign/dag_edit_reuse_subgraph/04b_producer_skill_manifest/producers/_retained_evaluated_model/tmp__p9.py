import numpy as np,pickle,importlib.util,itertools
from scipy.optimize import least_squares
sp=importlib.util.spec_from_file_location('f','/holdout_task/_frozen_nodes/node_00.py'); f=importlib.util.module_from_spec(sp);sp.loader.exec_module(f)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; out=f.run_node({'__input__':s}); tr=s['train']; y=tr['y']; T=tr['traj']; t=s['eval_times'];
def poly(x,p):
 terms=[i for d in range(p+1) for i in itertools.combinations_with_replacement(range(x.shape[1]),d)];return np.stack([np.prod(x[:,i],axis=1) for i in terms],1),terms
def ls(a,b):return np.linalg.lstsq(a,b,rcond=None)[0]
def err(a,b):return np.mean(abs(a-b))
u=s['panel_b']['u'];v=s['panel_b']['v'];a=f._affine(u); z=out['z']; zt=out['zt']; raw=np.linalg.solve(ls(a,v)[1:].T,(z-ls(a,v)[0]).T).T
from scipy.optimize import minimize_scalar
from scipy.special import erf
from scipy.linalg import svd
print('singular',svd(T,compute_uv=False).round(4)); print('t',t)
fun=[lambda x,h:np.sin(h*x),lambda x,h:np.tanh(h*x),lambda x,h:x*np.exp(-(h*x)**2),lambda x,h:x/(1+(h*x)**2),lambda x,h:np.sign(x)*abs(x)**h,lambda x,h:x**3/(1+(h*x)**2)]
for i,F in enumerate(fun):
 for k in range(3):
  def D(x,h):return np.c_[f._affine(x),F(x,h), *([x*np.roll(x,-1,1)] if k==1 else [x**2] if k==2 else [])]
  def opt(h):
   A=D(u,h);return np.linalg.norm(A@ls(A,v)-v)
  p=minimize_scalar(opt,bounds=(.15,4),method='bounded');print('panel',i,k,round(p.x,4),round(p.fun,5))
X=np.load('/tmp/aligned.npz')['X']; Q=poly(X,2)[0]; A=f._affine(X)
for mode in range(4):
 def fit_time(row):
  def M(k):
   e=-np.expm1(-k*t)/k
   return [e[:,None],np.c_[e,t],np.c_[e,t**2],np.c_[e,np.ones(len(t))]][mode]
  def obj(k):
   d=M(k);return np.linalg.norm(d@ls(d,row)-row)
  a=minimize_scalar(obj,bounds=(.005,5),method='bounded'); return np.r_[a.x,ls(M(a.x),row),a.fun]
 W=np.array([fit_time(row) for row in T]);print('tempor',mode,np.quantile(W,[.1,.5,.9],axis=0).round(4)); print('aff/std', (np.std(W-A@ls(A,W),axis=0)/np.std(W,axis=0)).round(3),'quad/std',(np.std(W-Q@ls(Q,W),axis=0)/np.std(W,axis=0)).round(3))
 if mode==0: print('coef',ls(Q,W).round(3))
np.savez('/tmp/probe9.npz',X=X,raw=raw,t=t,T=T,y=y)
# Fit a saturating or rational single-index component in addition to a quadratic.
for nm, F in [('sin',np.sin),('tanh',np.tanh),('rat',lambda a:1/(1+a*a)),('odd',lambda a:a/(1+a*a)),('cube',lambda a:a**3),('even',lambda a:np.log1p(a*a))]:
 for lab,Y in [('y',y),('traj',T)]:
  def D(p):return np.c_[Q,F(A@p)]
  def res(p):
   G=D(p)[:52];return (G@ls(G,Y[:52])-Y[:52]).ravel()
  rng=np.random.default_rng(21);starts=np.r_[np.eye(5),rng.normal(size=(22,5))]; sol=min([least_squares(res,p,max_nfev=100) for p in starts],key=lambda q:q.cost); G=D(sol.x); E=abs(G@ls(G[:52],Y[:52])-Y); print(nm,lab,sol.x.round(3),E[:52].mean().round(5),E[52:].mean().round(5))
