import numpy as np,pickle,runpy,glob
from itertools import combinations_with_replacement
from scipy.spatial.distance import cdist
st=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];tr=st['train'];x=tr['x'];Y=np.c_[tr['y'],tr['traj']]; sparse=runpy.run_path('/holdout_task/_frozen_nodes/node_00.py')['run_node']; arc=runpy.run_path(glob.glob('/tmp/hegel_continual_archive/*solver.py')[0]);
folds=[]
for b in [np.arange(52,72),np.arange(20),np.arange(20,40),np.arange(40,60),*np.array_split(np.random.default_rng(443).permutation(len(x)),4)]:
 a=np.setdiff1d(np.arange(len(x)),b);s=dict(st,train={k:v[a] for k,v in tr.items() if k!='summary'},test={'x':x[b]});d=sparse({'__input__':s});folds.append((a,b,np.c_[d['y_hat'],d['state_pred']]))
w=np.r_[.5,np.repeat(.65,8)]
def score(pred):
 out=[]
 for a,b,sp in folds:
  v=np.abs(w*pred(a,b)+(1-w)*sp-Y[b]);out.append([v[:,0].mean(),v[:,1:].mean(),v[:,1:].mean(0).max()]);
 return np.array(out)
def pol(x,deg):return np.column_stack([np.prod(x[:,c],axis=1) for d in range(1,deg+1) for c in combinations_with_replacement(range(4),d)])
def ridge(f,lam):
 def p(a,b):
  u=(f-f[a].mean(0))/np.maximum(f[a].std(0),1e-10); return Y[a].mean(0)+u[b]@u[a].T@np.linalg.solve(u[a]@u[a].T+lam*np.eye(len(a)),Y[a]-Y[a].mean(0))
 return p
res=[]
def append(name,p):
 s=score(p);res.append((max(s[:,:2].max(0)),name,s))
for deg in range(1,6):
 for l in (1e-8,1e-4,.003,.03,.3,1):append(('poly',deg,l),ridge(pol(x,deg),l))
d=cdist(x,x,'sqeuclidean')
for scale in (.08,.18,.35,.65,1.3,3,8):
 for lam in (1e-7,1e-5,.0003,.003,.03):
  for power in (0,1,2):
   t=np.sqrt(d*scale); K=(np.exp(-t*t) if power==0 else (1+t+(t*t/3 if power==2 else 0))*np.exp(-t))
   def p(a,b):return Y[a].mean(0)+K[np.ix_(b,a)]@np.linalg.solve(K[np.ix_(a,a)]+lam*np.eye(len(a)),Y[a]-Y[a].mean(0))
   append(('kernel',scale,lam,power),p)
for v in sorted(res,key=lambda v:v[0])[:14]:print(v[1],v[2].round(4).tolist())
for v in sorted(res,key=lambda v:v[2][0,:2].max())[:5]:print('gate best',v[1],v[2].round(4).tolist())
