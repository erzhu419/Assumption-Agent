import sys,pickle,itertools,numpy as np
sys.path.insert(0,'/holdout_task')
from _frozen_nodes.node_00 import run_node
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; z=run_node({'__input__':s})['z']; y=s['train']['y']; tr=s['train']['traj']; n,d=z.shape
np.set_printoptions(precision=5,suppress=True)
print('range',z.min(0),z.max(0),'target',y.min(),y.max()); a,b,c,e=z.T
base=np.column_stack([z,a*c+b*e]); aff=np.column_stack([np.ones(n),z]); rng=np.random.default_rng(282); folds=np.array_split(rng.permutation(n),6)
def score(X,Y,pen=0):
 pred=np.zeros_like(Y,dtype=float); vals=[]
 for te in folds:
  i=np.setdiff1d(np.arange(n),te); xx=X[i]; yy=Y[i]; co=np.linalg.lstsq(xx,yy,rcond=None)[0] if pen==0 else np.linalg.solve(xx.T@xx+pen*np.eye(X.shape[1]),xx.T@yy); pred[te]=X[te]@co; vals.append(np.mean(abs(pred[te]-Y[te])))
 co=np.linalg.lstsq(X,Y,rcond=None)[0]; return np.mean(abs(pred-Y)),max(vals),np.mean(abs(X@co-Y))
for name,X in [('lin',z),('aff',aff),('bilin',base),('affbilin',np.column_stack([np.ones(n),base]))]: print(name,score(X,y),score(X,tr))
for deg in range(2,5):
 ix=[i for k in range(1,deg+1) for i in itertools.combinations_with_replacement(range(d),k)]; X=np.column_stack([np.prod(z[:,i],axis=1) for i in ix]); print('poly',deg,[score(X,y,p) for p in (0,.0001,.001,.01,.1)]); print('traj poly',deg,score(X,tr,.01))
co=np.linalg.lstsq(base,y,rcond=None)[0]; r=y-base@co; print('base coef',co,'res range',r.min(),r.max()); print('traj sv',np.linalg.svd(tr,compute_uv=False)); print('traj residual sv',np.linalg.svd(tr-z@np.linalg.lstsq(z,tr,rcond=None)[0],compute_uv=False)); print('corr xy',np.corrcoef(np.column_stack((z,y,r)).T))
features={}
for k in range(1,5):
 for i in itertools.combinations_with_replacement(range(d),k): features[str(i)]=np.prod(z[:,i],axis=1)
for i,j in itertools.combinations(range(d),2):
 for sign in [-1,1]:
  for fun in [np.abs,np.sin,np.cos,np.tanh]: features[f'{fun.__name__}({i},{sign},{j})']=fun(z[:,i]+sign*z[:,j]);
for i in range(d):
 for f in [np.abs,np.sin,np.cos,np.tanh,np.exp]: features[f'{f.__name__}({i})']=f(z[:,i]);
for name,X in [('base+1',np.column_stack([np.ones(n),base])),('base',base)]:
 scores=sorted((score(np.column_stack((X,v)),y),k) for k,v in features.items()); print(name,scores[:18])
print('largest residual rows',np.column_stack([z,y,r])[np.argsort(abs(r))[-14:]])
