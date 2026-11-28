import numpy as np,pickle
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def load(p):
 d={};exec(open('/holdout_task/'+p+'.py').read(),d);return d
A=load('_frozen_nodes/node_02');B=load('_frozen_nodes/node_00');p=B['run_node']({'node_02':A['run_node']({'__input__':s})});z,b=[p[k][:72] for k in ('z','b')];u=s['panel_c']['u'];v1=s['panel_c']['v1'];v2=s['panel_c']['v2']; y=s['train']['y'];tr=s['train']['traj'];t=s['eval_times'];
def err(X,Y):
 X=np.column_stack((np.ones(len(X)),X));w=np.linalg.lstsq(X,Y,rcond=None)[0];return np.sqrt(np.mean((X@w-Y)**2,axis=0))
if __name__=='__main__':
 np.set_printoptions(precision=5,suppress=True);print('ranges',np.min(z,0),np.max(z,0),np.min(b,0),np.max(b,0));print('trajectory\n',tr[:8]);print('svd',np.linalg.svd(tr,compute_uv=False), 'diff ratios',np.diff(tr[:8],n=2)/np.diff(tr[:8],axis=1)[:,:-1]);print('linear');
 for f in [z,b,np.c_[z,b],np.c_[z,b,z*z,b*b,z*b]]:print(err(f,np.c_[y,tr]));
 for name,Y in [('v1',v1),('v2',v2)]:
  print(name,'ranges',Y.min(0),Y.max(0));print('coef',np.linalg.lstsq(np.c_[np.ones(len(u)),u],Y,rcond=None)[0]);
  for f in [u,np.c_[u,u*u],np.c_[u,*[u[:,i,None]*u[:,i:] for i in range(4)]],np.c_[u,np.tanh(3.5*u),u/(1+6*u*u)],np.c_[u,np.sin(3.5*u),np.cos(3.5*u)]]:print(err(f,Y));
  print(np.round(np.c_[u,Y],3))
