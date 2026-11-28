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
from scipy.linalg import qr
from scipy.special import erf
Y=np.c_[y,traj@np.linalg.svd(traj,full_matrices=False)[2][:2].T]; Y/=Y.std(0)
from collections import OrderedDict
d=OrderedDict()
def put(a,k):
 for i,c in enumerate(a.T): d[f'{k}{i}']=c
put(z,'z');put(b,'b');put(np.cos(2.6*z),'c');
for k,x in [('z',z),('b',b),('c',np.cos(2.6*z))]:
 for q in [.5,1,1.5,2,2.5,3,4,5]:
  put(np.sin(q*x),f'sin{q}{k}');put(np.cos(q*x),f'cos{q}{k}');put(np.tanh(q*x),f'tanh{q}{k}');put(x/(1+q*x*x),f'rat{q}{k}');put(np.exp(-q*x*x),f'gauss{q}{k}')
 for q in [2,3,4]:put(x**q,f'{k}^{q}:')
 for i,j in itertools.combinations(range(4),2):
  d[f'{k}{i}*{k}{j}']=x[:,i]*x[:,j]
  for a,g in [(1,x[:,i]+x[:,j]),(-1,x[:,i]-x[:,j])]:
   for q in [1,2,3]:
    for fn in [np.sin,np.cos,np.tanh]:d[f'{fn.__name__}{q}({k}{i}{a:+}{k}{j})']=fn(q*g)
for i,j in itertools.product(range(4),repeat=2):
 d[f'z{i}b{j}']=z[:,i]*b[:,j];d[f'z{i}c{j}']=z[:,i]*np.cos(2.6*z[:,j]);d[f'z{i}tb{j}']=z[:,i]*np.tanh(2.5*b[:,j]);d[f'tz{i}b{j}']=np.tanh(z[:,i])*b[:,j]
D=np.column_stack(list(d.values())); names=np.array(list(d));
from sklearn.linear_model import lars_path, ElasticNet
for j in range(3):
 for sel in [folds[0],folds[1]]:
  i,v=sel; mu=D[i].mean(0); sd=D[i].std(0); A=(D-mu)/np.maximum(sd,1e-8); m=Y[i,j].mean(); _,inds,co=lars_path(A[i],Y[i,j]-m,method='lasso',max_iter=160); err=np.mean(abs(A[v]@co+m-Y[v,j,None]),axis=0); k=np.argmin(err); print('LARS',j,k,err[k],[(names[h],round(co[h,k],4)) for h in np.argsort(-abs(co[:,k]))[:12]]); print('corr',[(names[h],round(np.mean(A[i,h]*(Y[i,j]-m)),3)) for h in np.argsort(-abs(A[i].T@(Y[i,j]-m)))[:12]])
