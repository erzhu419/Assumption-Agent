import numpy as np,pickle
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=5,suppress=True)
def ls(f,y):return np.linalg.lstsq(f,y,rcond=None)[0]
def aff(x):return np.c_[np.ones(len(x)),x]
for name in ('a','b','c'):
 d=s['panel_'+name]; u=d['u'];
 for key in set(d)-{'u'}:
  v=d[key]; k=np.linalg.norm(v-u,axis=1)<4 if name=='a' else np.ones(len(u),bool); x,y=u[k],v[k]; print(name,key,len(x))
  for label,f in [('linear',aff(x)),('poly2',np.c_[aff(x),*[x[:,i]*x[:,j] for i in range(4) for j in range(i,4)]]),('sin',np.c_[aff(x),np.sin(2.4*x)]),('tanh',aff(np.tanh(2.15*x))),('atan',aff(np.arctan(2.15*x)))]:
   w=ls(f,y); print(label,np.sqrt(np.mean((f@w-y)**2)), np.sqrt(np.mean(((f@w-y)/(1-np.sum(np.linalg.svd(f,full_matrices=False)[0]**2,axis=1))[:,None])**2))); 
   if label=='linear': print(w)
z=s['train']['traj']; y=s['train']['y']; t=s['eval_times']; print('t',t); p,d,v=np.linalg.svd(z,full_matrices=False);print('svd',d); print('modes',v[:5]); print('y from z',ls(aff(z),y),np.mean(abs(aff(z)@ls(aff(z),y)-y))); print('first differences SVD',np.linalg.svd(np.diff(z),compute_uv=False)); print('z residual rank 2', np.mean(abs(z-(p[:,:2]*d[:2])@v[:2]),axis=0));
