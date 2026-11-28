import numpy as np,pickle,itertools,runpy
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; n=52
m=runpy.run_path('/holdout_task/_frozen_nodes/node_00.py'); o=m['run_node']({'__input__':s}); b=lambda x:m['_b_features'](x,o['p'])@o['w']; print('operator',o)
q=(s['train']['x']-o['offset'])/o['gain']; y=s['train']['y']; t=s['train']['traj']; u=s['panel_c']['u']; v1=s['panel_c']['v1']; v2=s['panel_c']['v2']; g=b(q); bu=b(u); print('svd t',np.linalg.svd(t,compute_uv=False)); print('t samples',t[:8]); print('svd C',np.linalg.svd(np.c_[v1,v2],compute_uv=False))
def ls(x,y):return np.linalg.lstsq(x,y,rcond=None)[0]
def f(*xs):return np.column_stack([np.ones(len(xs[0])),*xs])
def poly(x,d):return np.column_stack([np.prod(x[:,idx],axis=1) for k in range(d+1) for idx in itertools.combinations_with_replacement(range(x.shape[1]),k)])
def pairs(x):return np.column_stack([x[:,i]*x[:,j] for i,j in itertools.combinations_with_replacement(range(x.shape[1]),2)])
def report(name,x,Y):
 pred=x[n:]@ls(x[:n],Y[:n]); print(name,x.shape[1],np.mean(abs(pred-Y[n:]),axis=0).round(4),np.mean(abs(pred-Y[n:])).round(5))
for name,a,c in [('u',u,q),('b(u)',bu,g),('u+b',np.c_[u,bu],np.c_[q,g]),('u+buquad',f(u,bu,pairs(bu)),f(q,g,pairs(g))),('uquad+b',f(u,bu,pairs(u)),f(q,g,pairs(q)))]:
 a=f(a); err=[]; extrap=[]
 for v in [v1,v2]:
  err.append(np.mean(abs(a@ls(a,v)-v))); extrap.append(f(c)@ls(a,v))
 print('C fits',name,err)
for name,x in [('q',q),('B',g),('q+B',np.c_[q,g])]:
 for d in range(1,4 if x.shape[1]==4 else 3):
  report(name+str(d)+'traj',poly(x,d),t);report(name+str(d)+'y',poly(x,d),y)
