import pickle,numpy as np
from itertools import combinations_with_replacement
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def ls(a,b):return np.linalg.lstsq(a,b,rcond=1e-10)[0]
u,v=[s['panel_a'][k] for k in 'uv'];m=np.max(abs(v-u),axis=1)<3;u=u[m];v=v[m]
libs={}
def score(a,y):
 w=ls(a,y); h=np.clip(1-np.sum(a*np.linalg.pinv(a).T,axis=1),1e-9,None);return np.mean(((a@w-y)/h)**2)**.5,w
for j in range(4):
 x=u[:,j];z=u[:,(j+1)%4];y=v[:,j];print('J',j)
 for key,a in [('bilin',np.c_[np.ones(len(u)),x,z,x*z]),('quad',np.c_[np.ones(len(u)),x,z,x*z,x*x,z*z]),('cubic',np.c_[np.ones(len(u)),x,z,x*z,x*x,z*z,x*x*z,x*z*z,x**3,z**3]),('qall',np.c_[np.ones(len(u)),u,*[u[:,i]*u[:,k] for i,k in combinations_with_replacement(range(4),2)]])]:print(key,score(a,y))
 a=np.c_[np.ones(len(u)),u]; candidates={}
 for i in range(4):
  for k in range(i,4): candidates[f'p{i}{k}']=u[:,i]*u[:,k]
 for i in range(4):
  for k in range(4):
   for h in [.22,.5,1.25]:
    candidates[f'r{i}{k}_{h}']=u[:,i]/(1+h*u[:,k]**2)
    candidates[f'b{i}{k}_{h}']=u[:,i]*u[:,k]/(1+h*u[:,k]**2)
 for _ in range(4):
  k=min(candidates,key=lambda k:score(np.c_[a,candidates[k]],y)[0]);a=np.c_[a,candidates.pop(k)]; print('add',k,score(a,y))
