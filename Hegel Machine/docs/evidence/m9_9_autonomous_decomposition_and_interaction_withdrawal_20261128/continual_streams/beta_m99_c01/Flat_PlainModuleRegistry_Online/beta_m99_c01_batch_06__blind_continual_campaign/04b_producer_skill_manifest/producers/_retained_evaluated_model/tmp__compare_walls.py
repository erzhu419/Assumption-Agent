import sys,pickle,json,importlib.util,itertools
import numpy as np
from scipy.sparse import diags,eye,kron
from scipy.sparse.linalg import splu
sys.path.insert(0,'/holdout_task')
from _verified_upstream import calibrate_upstream
spec=importlib.util.spec_from_file_location('prev','/tmp/hegel_continual_archive/beta_m99_c01_batch_05_solver.py'); prev=importlib.util.module_from_spec(spec); spec.loader.exec_module(prev)
s=pickle.load(open('public_batch.pkl','rb'))[0]['study']; y=calibrate_upstream(s,json.load(open('_upstream_param_payload.json')))['y_obs_raw']; yfit=calibrate_upstream(s)['y_obs_raw']; print('dynamic-vs-payload max obs difference',np.max(np.abs(yfit-y))); y=yfit
x,yidx=s['sensor_ij'].T; nx,ny=s['nx'],s['ny']; z=s['Z_nuisance']; eyez=np.eye(z.shape[1]); ridge=s['source_ridge']; cap=s['u_box']; budget=s['total_budget']; mus=s['mu_shifts']; modes=s['source_modes']; rhs=modes.transpose(0,2,1).reshape(len(modes),-1).T

def axis(name,kind):
 k,v=s['kappa_'+name],s['vel_'+name]; h=s['L'+name]/(len(k)-1)
 if kind=='P': return prev._axis(k,v,h,True)
 a=prev._axis(k,v,h,False).tolil()
 if kind=='O':
  a[0,0]+=s['robin_'+name]/h; a[-1,-1]+=s['robin_'+name]/h
 return a.tocsc()

def fit(a,train):
 d=np.column_stack((a,y)); p=np.linalg.solve(z[train].T@z[train]+s['nuisance_ridge']*eyez,z[train].T@d[train]); At,yt=(d[train]-z[train]@p)[:,:-1],(d[train]-z[train]@p)[:,-1]; w,nu=prev._box_simplex(At.T@At+ridge*np.eye(len(modes)),-At.T@yt,cap,budget); return w,p[:,-1]-p[:,:-1]@w
for a,b in itertools.product('ORP',repeat=2):
 l=kron(eye(ny),axis('x',a))+kron(axis('y',b),eye(nx))+diags(s['V_reaction_2d'].ravel(order='F')); A=np.concatenate([splu((l+m*eye(nx*ny)).tocsc()).solve(rhs)[x+nx*yidx] for m in mus]); w,g=fit(A,np.ones(len(y),bool)); r=A@w+z@g-y
 rr=np.zeros_like(y)
 for i in range(len(x)):
  test=np.arange(len(y))%len(x)==i; wi,gi=fit(A,~test); rr[test]=A[test]@wi+z[test]@gi-y[test]
 print(a+b,'rmse %.7f loso %.7f boundary %.7f worst shift %.7f'% (np.sqrt(np.mean(r*r)),np.sqrt(np.mean(rr*rr)),np.sqrt(np.mean(rr.reshape(len(mus),-1)[:,(x==0)|(x==nx-1)|(yidx==0)|(yidx==ny-1)]**2)),np.max(np.sqrt(np.mean(rr.reshape(len(mus),-1)**2,axis=1)))), 'w',np.round(w,4))
