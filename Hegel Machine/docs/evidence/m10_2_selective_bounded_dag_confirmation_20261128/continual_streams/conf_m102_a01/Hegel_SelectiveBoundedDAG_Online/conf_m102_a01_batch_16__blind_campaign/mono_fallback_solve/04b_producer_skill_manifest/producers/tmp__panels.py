import numpy as np,pickle,importlib.util
from itertools import combinations_with_replacement
S=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; X=S['train']['x']; Y=S['train']['y']; T=S['train']['traj']; n=len(X)
p='/tmp/hegel_continual_archive/conf_m102_a01_batch_03_solver.py'; spec=importlib.util.spec_from_file_location('old',p); old=importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
A=lambda x: np.column_stack([np.ones(len(x)),x]); mae=lambda x: np.mean(abs(x),axis=0)
np.set_printoptions(precision=5,suppress=True,linewidth=150)
from scipy.optimize import minimize_scalar
p=S['panel_a']; D=A(p['u']); v=p['v']; ids=np.arange(len(D))
for _ in range(6):
 b=np.linalg.lstsq(D[ids],v[ids],rcond=None)[0]; e=np.linalg.norm(D@b-v,axis=1); ids=np.argsort(e)[:-3]
print('panel A cleaned',b,'RMSE retained',np.sqrt(np.mean(e[ids]**2)), 'omitted',np.argsort(e)[-3:]); print('singular',np.linalg.svd(b[1:],compute_uv=False)); print('diag fits',old._coordinates(S,X)[0][:3])
p=S['panel_c']; u=p['u'];
for k in ['v1','v2']:
 v=p[k]; print(k,'singular affine',np.linalg.svd(np.linalg.lstsq(A(u),v,rcond=None)[0][1:],compute_uv=False))
 for a in [.22,1.25]:
  for name,fun in [('tanh',lambda z: np.tanh(a*z)),('rat',lambda z:z/(1+a*z*z)),('sin',lambda z:np.sin(a*z))]:
   D=A(fun(u)); pred=D@np.linalg.lstsq(D,v,rcond=None)[0]; print(name,a,np.sqrt(np.mean((pred-v)**2)))
 D=np.column_stack([np.ones(len(u))]+[np.prod(u[:,i],axis=1) for deg in [1,2] for i in combinations_with_replacement(range(4),deg)]); print('quadratic',np.sqrt(np.mean((D@np.linalg.lstsq(D,v,rcond=None)[0]-v)**2)))
 if k=='v2':
  D=np.c_[A(u),p['v1'],p['v1']**2,p['v1']*u]; print('joint',np.sqrt(np.mean((D@np.linalg.lstsq(D,v,rcond=None)[0]-v)**2)))
t=S['eval_times']; shape=lambda r: -np.expm1(-r*t); err=lambda r: np.mean((T-(T@shape(r))[:,None]*shape(r)/(shape(r)@shape(r)))**2); fit=minimize_scalar(err,bounds=(.05,3),method='bounded'); print('time exp fit',fit.x,np.sqrt(fit.fun),np.sqrt(err(.9)))
