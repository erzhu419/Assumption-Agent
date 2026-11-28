import sys,pickle,numpy as np
from scipy.optimize import minimize_scalar
from itertools import combinations,combinations_with_replacement
sys.path[:0]=['/holdout_task','/holdout_task/_frozen_nodes']; import node_00
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=5,suppress=True)
def one(x):return np.c_[np.ones(len(x)),x]
def fit(h,v):return np.linalg.lstsq(h,v,rcond=None)[0]
def quad(z):return np.column_stack([z[:,i]*z[:,j] for i,j in combinations_with_replacement(range(z.shape[1]),2)])
def profile(u,v,keep,fn,bounds):
 def err(w):
  h=fn(u,w); a=fit(h[keep],v[keep]);return np.mean((h[keep]@a-v[keep])**2)
 p=minimize_scalar(err,bounds=bounds,method='bounded');return np.sqrt(p.fun),p.x,fit(fn(u,p.x)[keep],v[keep])
u,v=[s['panel_a'][k] for k in ['u','v']]; h=one(u); r=np.linalg.norm(v-h@fit(h,v),axis=1); keep=np.argsort(r)[:45];print('A keep',sorted(set(range(len(u)))-set(keep)))
for name,fn in [('linear+tanh',lambda z,w:np.c_[one(z),np.tanh(w*z)]),('linear+sin',lambda z,w:np.c_[one(z),np.sin(w*z)]),('linear + tanh + quad',lambda z,w:np.c_[one(z),np.tanh(w*z),quad(z)]),('tanh sums',lambda z,w:np.c_[one(z),np.tanh(w*(z+np.roll(z,2,axis=1)))]),('tanh diffs',lambda z,w:np.c_[one(z),np.tanh(w*(z-np.roll(z,2,axis=1)))])]:
 e,w,a=profile(u,v,keep,fn,(.08,4)); print('A',name,e,w); print(a if name=='linear+tanh' else '')
u=s['panel_c']['u'];
for k in ['v1','v2']:
 v=s['panel_c'][k]
 for name,fn in [('tanh',lambda z,w:one(np.tanh(w*z))),('lin+tanh',lambda z,w:np.c_[one(z),np.tanh(w*z)]),('lin+sin',lambda z,w:np.c_[one(z),np.sin(w*z)]),('lin+gauss',lambda z,w:np.c_[one(z),np.expm1(-w*z*z)]),('tanh sums',lambda z,w:np.c_[one(z),np.tanh(w*(z+np.roll(z,1,axis=1)))]),('tanh diffs',lambda z,w:np.c_[one(z),np.tanh(w*(z-np.roll(z,1,axis=1)))])]:
  e,w,a=profile(u,v,np.arange(len(u)),fn,(.05,5)); print('C',k,name,e,w); print(a if e<.003 else '')
tr=s['train']; x=tr['x']; idx=np.arange(len(x)); va=idx[52:]; it=idx[:52]; st=dict(s,train={k:tr[k][it] for k in ('x','y','traj')},test={'x':x[va]}); sp=node_00.run_node({'__input__':st}); print('frozen alone',np.mean(abs(sp['y_hat']-tr['y'][va])),np.mean(abs(sp['state_pred']-tr['traj'][va])))
