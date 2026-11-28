import sys,pickle,itertools,numpy as np
sys.path.insert(0,'/holdout_task')
from node_00 import _latent,_panel_b
from _frozen_nodes.node_01 import _kernel
from _frozen_nodes.node_02 import run_node as sink
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; y=t['y']; traj=t['traj']; n=len(y)
z=_latent(s['panel_a'],t['x']); b=_panel_b(s['panel_b'],z); print('n',n,'z ranges',z.min(0),z.max(0),'b ranges',b.min(0),b.max(0))
rng=np.random.default_rng(214); folds=[(np.arange(max(40,n-20)),np.arange(max(40,n-20),n))]; folds += [(np.setdiff1d(np.arange(n),v),v) for v in np.array_split(rng.permutation(n),4)]
def score(x):
 out=[]
 for i,j in folds:
  a=x[i]; c=x[j]; p=sink({'__input__':{'train':{'y':y[i],'traj':traj[i]}},'node_01':{'k':_kernel(a,a),'ktest':_kernel(c,a)}})
  out.append([np.abs(p['y_hat']-y[j]).mean(),np.abs(p['state_pred']-traj[j]).mean()])
 return np.array(out)
def rank(a):return max(a.max(0))+.2*a.mean()
groups={'z':z,'b':b,'tz':np.tanh(z),'tb':np.tanh(2.5*b),'c':np.cos(2.6*z),'si':np.sin(2.6*z),'s':np.sin(z),'p':z**2,'q':b**2}
from scipy.linalg import solve
from scipy.special import erf
from scipy.optimize import minimize
Y=np.c_[y,traj]; ys=np.r_[np.std(y),np.repeat(np.std(traj),traj.shape[1])]; T=Y/ys
C=np.cos(2.60016*z)
from numpy.polynomial.legendre import legvander
def kern(F):
 F=F-F.mean(0); F=F/np.maximum(F.std(0),1e-7); return F@F.T/F.shape[1]
def add(F): return sum(kern(f) for f in F)/len(F)
K={}
for name,X in [('z',z),('b',b),('c',C)]:
 K[name]=kern(X)
 for q in [1.,1.5,2.,2.5,3.,4.]:
  K[f'{name}t{q}']=kern(np.tanh(q*X)); K[f'{name}s{q}']=kern(np.sin(q*X)); K[f'{name}e{q}']=kern(erf(q*X))
 for p in [2,3,4,5,6]: K[f'{name}^{p}']=kern(X**p)
 K[name+'diag']=add([np.c_[np.sin(2*X[:,i]),np.sin(4*X[:,i]),np.cos(2*X[:,i]),np.cos(4*X[:,i])] for i in range(4)])
for a,p in [('z',z),('b',b),('c',C),('t',np.tanh(2*b)),('s',np.sin(2.6*z))]:
 for d,q in [('z',z),('b',b),('c',C),('t',np.tanh(2*b)),('s',np.sin(2.6*z))]:
  if a>d:continue
  K[a+d]=kern(np.column_stack([p[:,i]*q[:,j] for i,j in itertools.product(range(4),repeat=2) if a!=d or i<j]))
 K[a+'coord']=kern(p)
 for j in range(4): K[f'{a}one{j}']=kern(p[:,j:j+1])
N=list(K); K=np.array(list(K.values())); nc=len(K)
def learned(i,j,pen):
 A=K[:,i][:,:,i].copy(); B=K[:,j][:,:,i].copy(); E=np.eye(len(i)); W=T[i].copy(); W[:,1:]/=np.sqrt(T.shape[1]-1)
 def obj(u):
  w=np.exp(u); G=(w@A.reshape(nc,-1)).reshape(E.shape)+1+0.004*E
  Q=solve(G,E,assume_a='pos'); alpha=Q@W; D=Q.diagonal(); R=alpha/D[:,None]
  da=2*R/D[:,None]; dq=-2*np.sum(R*R,1)/D; dg=-(Q@da)@alpha.T-(Q*dq)@Q
  return np.sum(R*R)+pen*np.sum(w), w*(A.reshape(nc,-1)@dg.ravel()+pen)
 opt=minimize(obj,np.full(nc,-4.),jac=True,bounds=[(-12,4)]*nc,method='L-BFGS-B',options={'maxiter':170}); w=np.exp(opt.x); G=(w@A.reshape(nc,-1)).reshape(E.shape)+1+0.004*E
 return (1+np.tensordot(w,B,axes=1))@solve(G,Y[i],assume_a='pos'),w
for pen in [.001,.015,.12,1.]:
 es=[]
 for j,(i,v) in enumerate(folds):
  p,w=learned(i,v,pen); e=np.abs(p-Y[v]);es.append([e[:,0].mean(),e[:,1:].mean()]);
  if j==0:print('selected',pen,[(N[k],round(w[k],4)) for k in np.argsort(-w)[:15]],flush=True)
 print('errors',pen,np.round(es,5).tolist(),flush=True)
