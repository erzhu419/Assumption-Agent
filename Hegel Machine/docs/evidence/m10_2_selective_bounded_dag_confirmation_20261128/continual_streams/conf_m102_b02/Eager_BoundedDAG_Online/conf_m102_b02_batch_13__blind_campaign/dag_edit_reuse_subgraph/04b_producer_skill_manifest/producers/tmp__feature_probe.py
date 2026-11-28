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
res=[]
for names in [('z','b','tz','tb'),('z','b'),('z','b','tz'),('z','b','tb'),('z','b','c'),('z','b','si'),('z','c'),('z','b','s'),('tz','b'),('z','tb'),('z','b','p','q'),('z','b','tz','tb','c'),('z','b','p')]:
 for std,center in itertools.product([False,True],repeat=2):
  xs=[(groups[k]-(groups[k].mean(0) if center else 0))/(groups[k].std(0) if std else 1) for k in names]
  for scale,rel in itertools.product([.7,1,1.5,2,3,5,8],[.5,1,2]):
   a=score(np.hstack([scale*x*(rel if k in ('b','tb','c','q') else 1) for x,k in zip(xs,names)]));res.append((rank(a),names,std,center,scale,rel,a))
for r in sorted(res,key=lambda x:x[0])[:16]:print(r[:-1],r[-1].round(5).tolist())
print('best official');
for r in sorted(res,key=lambda x:max(x[-1][0]))[:6]: print(r[:-1],r[-1].round(5).tolist())
print('linear fit diagnostics');
for keys in [('z','b'),('z','b','tz','tb'),('z','c'),('z','c','si'),('z','b','p','q')]:
 x=np.hstack([groups[k] for k in keys]);
 for deg in [1,2]:
  a=np.column_stack([np.ones(n), x]+([x[:,i]*x[:,j] for i in range(x.shape[1]) for j in range(i,x.shape[1])] if deg==2 else [])); i,j=folds[0]; co=np.linalg.lstsq(a[i],np.c_[y,traj[:,-1]][i],rcond=1e-7)[0]; print(keys,deg,a.shape, np.abs(a@co-np.c_[y,traj[:,-1]])[i].mean(0),np.abs(a@co-np.c_[y,traj[:,-1]])[j].mean(0))
