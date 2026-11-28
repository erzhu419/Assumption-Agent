import sys,pickle,itertools as it
import numpy as np
sys.path.insert(0,'/holdout_task/_frozen_nodes')
from node_00 import run_node
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; q=run_node({'__input__':s})['q_tr']; y=s['train']['y']; z=s['train']['traj']; t=s['eval_times']; c=s['panel_c']; u=c['u']
np.set_printoptions(precision=5,suppress=True)
def F(x,*cols): return np.column_stack([np.ones(len(x)),*cols])
def loo(X,y):
 p=np.linalg.pinv(X); e=y-X@(p@y); return np.mean(np.abs((e.T/(1-np.sum(X*p.T,axis=1))).T),axis=0)
def fit(X,y):return np.linalg.lstsq(X,y,rcond=None)[0]
print('times',t,'telemetry',s['telemetry_vec'],'traj singular values',np.linalg.svd(z,compute_uv=False))
print('panel c u range',u.min(0),u.max(0),'q range',q.min(0),q.max(0));print('panel c first 6:',np.c_[u,c['v1'],c['v2']][:6])
for name,X in [('lin',F(u,u)),('square',F(u,u,u*u)),('cube',F(u,u,u**3)),('sq-cube',F(u,u,u*u,u**3)),('sin2',F(u,u,np.sin(2*u))),('tanh2',F(u,u,np.tanh(2*u))),('diag-quartic',F(u,u,u*u,u**4))]:
 print(name,loo(X,c['v1']),loo(X,c['v2']))
 def cv(X,Y):
  e=abs(X[52:]@fit(X[:52],Y[:52])-Y[52:]); return np.mean(e),np.max(np.mean(e,axis=0))
for name,x in [('q',q),('raw',s['train']['x'])]:
 for deg in (1,2,3):
  X=F(x,*[np.prod(x[:,inds],axis=1) for d in range(1,deg+1) for inds in it.combinations_with_replacement(range(4),d)]);print(name,deg,'cv y/traj',cv(X,y),cv(X,z))
 for tag,xx in [('diag2',F(x,x,x*x)),('diag3',F(x,x,x*x,x**3)),('diag4',F(x,x,x*x,x**3,x**4)),('tanh',F(x,x,np.tanh(1.7*x),x*x)),('sin',F(x,x,np.sin(1.7*x),x*x))]:print(name,tag,cv(xx,y),cv(xx,z))
