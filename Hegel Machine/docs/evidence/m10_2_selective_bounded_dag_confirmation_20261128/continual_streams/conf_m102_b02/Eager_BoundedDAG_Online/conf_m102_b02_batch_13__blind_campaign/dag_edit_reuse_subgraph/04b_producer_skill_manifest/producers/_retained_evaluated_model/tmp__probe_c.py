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
from scipy.special import erf
from scipy.optimize import minimize_scalar
p=s['panel_c'];u=p['u'];print('C', {k:(v.shape,v.min(),v.max()) for k,v in p.items()});
def L(x):return np.c_[np.ones(len(x)),x]
def fit(a,v):return np.linalg.lstsq(a,v,rcond=1e-9)[0]
cyc=lambda x:x*np.roll(x,-1,axis=1)
allp=lambda x:np.stack([x[:,i]*x[:,j] for i in range(x.shape[1]) for j in range(i,x.shape[1])],axis=1)
func={'id':lambda x,q:x,'tanh':lambda x,q:np.tanh(q*x),'sin':lambda x,q:np.sin(q*x),'erf':lambda x,q:erf(q*x),'cos':lambda x,q:np.cos(q*x),'rat':lambda x,q:x/(1+q*x*x),'soft':lambda x,q:x/np.sqrt(1+q*x*x),'x3':lambda x,q:np.c_[x,x**3],'cyc':lambda x,q:np.c_[x,cyc(x)],'roll':lambda x,q:np.c_[x,cyc(x),x**2], 'quad':lambda x,q:np.c_[x,allp(x)], 'id+tanh':lambda x,q:np.c_[x,np.tanh(q*x)], 'id+sin':lambda x,q:np.c_[x,np.sin(q*x)], 'tanh+cyc':lambda x,q:np.c_[np.tanh(q*x),cyc(x)], 'sin+cyc':lambda x,q:np.c_[np.sin(q*x),cyc(x)]}
for name in ('v1','v2'):
 v=p[name]; res=[]
 for k,f in func.items():
  def err(q,ret=False):
   a=L(f(u,q));w=fit(a,v);e=a@w-v
   if not ret:return np.mean(e**2)
   return np.sqrt(np.mean((e/(1-(a*np.linalg.pinv(a).T).sum(1))[:,None])**2)),w
  grid=np.linspace(.1,5,70);i=np.argmin([err(q) for q in grid]); q=minimize_scalar(err,bounds=(grid[max(0,i-1)],grid[min(len(grid)-1,i+1)]),method='bounded').x
  cv,w=err(q,True);res.append((cv,np.sqrt(err(q)),k,q,w))
 for cv,r,k,q,w in sorted(res,key=lambda a:a[0])[:8]: print(name,k,round(q,5),r,cv);groups[name+'_'+k]=L(func[k](z,q))@w
 groups[name]=groups[name+'_'+min(res,key=lambda a:a[0])[2]]
print('A residual');a=s['panel_a'];e=_latent(a,a['v'])-a['u'];print(np.quantile(np.linalg.norm(e,axis=1),[0,.5,.8,.9,.95,1]));print('train keys',list(t));
res=[]
for names in [('z','b','v1'),('z','b','v2'),('z','b','v1','v2'),('b','v1'),('b','v2'),('z','v1','v2'),('z','b','tb','v1','v2'),('z','b','tz','tb','v1','v2'),('z','b','tb','v1'),('z','b','tb','v2')]:
 for std,center in itertools.product([False,True],repeat=2):
  xs=[(groups[k]-(groups[k].mean(0) if center else 0))/(groups[k].std(0) if std else 1) for k in names]
  for scale,rel,cs in itertools.product([.5,1,1.5,2,3],[.5,1],[.5,1,2]):
   a=score(np.hstack([scale*x*(rel if k in ('b','tb') else cs if k in ('v1','v2') else 1) for x,k in zip(xs,names)]));res.append((rank(a),names,std,center,scale,rel,cs,a))
for r in sorted(res,key=lambda x:x[0])[:8]+sorted(res,key=lambda x:max(x[-1][0]))[:6]: print(r[:-1],r[-1].round(5).tolist())
