import numpy as np,pickle,importlib.util,copy
from scipy.special import erf
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
np.set_printoptions(precision=4,suppress=True,linewidth=130)
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
st=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
old=load('old','/tmp/hegel_continual_archive/conf_m102_b02_batch_01_solver.py')
for pan,keys in [('a',['v']),('b',['v']),('c',['v1','v2'])]:
 p=st['panel_'+pan];u=p['u']
 for k in keys:
  v=p[k];print(pan,k,'ranges',np.min(u,0),np.max(u,0),np.min(v,0),np.max(v,0),'sv',np.linalg.svd(v-v.mean(0),compute_uv=False))
  if pan=='c':print(np.hstack((u,v)))
  for d in [u,np.c_[u,u**3],np.c_[u,u**2,u**3],np.c_[u,u*abs(u),u**3],np.column_stack([np.prod(u[:,i],axis=1) for n in [1,2] for i in combinations_with_replacement(range(4),n)])]:
   d=np.c_[np.ones(len(u)),d];w=np.linalg.lstsq(d,v,rcond=None)[0];print('linear gram',d.shape[1],np.sqrt(np.mean((d@w-v)**2)),w.T[:,:5])
  if pan!='a':
   z=old._operator(p,k,u,pan);print('old resid',np.sqrt(np.mean((z-v)**2)))
   if pan=='c':
    print('old residual sv',np.linalg.svd(z-v,compute_uv=False))
s=copy.deepcopy(st);n=52;s['train']={k:v[:n] if k!='summary' else v for k,v in st['train'].items()};s['test']={'x':st['train']['x'][n:]};o=old.solve_batch(s);print('BASELINE',abs(o['y_hat']-st['train']['y'][n:]).mean(),abs(o['state_pred']-st['train']['traj'][n:]).mean())
