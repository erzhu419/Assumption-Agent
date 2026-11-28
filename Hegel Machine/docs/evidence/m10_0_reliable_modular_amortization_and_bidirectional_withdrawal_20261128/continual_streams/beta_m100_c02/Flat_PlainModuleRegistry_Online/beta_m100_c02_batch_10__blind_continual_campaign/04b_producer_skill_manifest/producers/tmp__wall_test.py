import sys,pickle,numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.optimize import minimize
sys.path.insert(0,'/holdout_task')
from _verified_upstream import calibrate_upstream
import json,itertools
p='/holdout_task/'
s0=pickle.load(open(p+'public_batch.pkl','rb'))[0]['study']
s=calibrate_upstream(s0,json.load(open(p+'_upstream_param_payload.json')))
y=s['y_obs_raw']; z=s['Z_nuisance']; n,m=s['nx'],s['ny']; k=len(s['source_modes']); q=s['sensor_ij']; inds=q[:,0]+n*q[:,1]; ids=np.tile(inds,len(s['mu_shifts'])); rhs=np.stack([v.ravel(order='F') for v in s['source_modes']],axis=1)
def axis(d,kind):
 a=s['kappa_'+d]; v=s['vel_'+d]; h=s['L'+d]/(len(a)-1); N=len(a); f=(a[:-1]+a[1:])/(2*h*h); M=np.zeros((N,N)); i=np.arange(1,N-1); M[i,i-1]=-f[:-1]-np.maximum(v[1:-1],0)/h; M[i,i+1]=-f[1:]-np.maximum(-v[1:-1],0)/h
 if kind=='W':
  seam=(a[0]+a[-1])/(2*h*h); M[0,1]=-f[0]-max(-v[0],0)/h; M[-1,-2]=-f[-1]-max(v[-1],0)/h; M[0,-1]=-seam-max(v[0],0)/h; M[-1,0]=-seam-max(-v[-1],0)/h
 else: M[0,1]=-a[0]/h**2; M[-1,-2]=-a[-1]/h**2
 M[np.diag_indices(N)]=-M.sum(1)
 if kind=='O': M[0,0]+=s['robin_'+d]/h; M[-1,-1]+=s['robin_'+d]/h
 return sparse.csc_matrix(M)
def response(ax,ay):
 L=sparse.kron(sparse.eye(m),ax,format='csc')+sparse.kron(ay,sparse.eye(n),format='csc')+sparse.diags(s['V_reaction_2d'].ravel(order='F'),format='csc'); eye=sparse.eye(n*m,format='csc'); return np.vstack([splu(L+mu*eye).solve(rhs)[inds] for mu in s['mu_shifts']])
def fit(S,tr,ridge=0):
 a=S[tr]; zz=z[tr]; yy=y[tr]; mat=np.column_stack((a,zz)); cap=s['u_box']; bud=s['total_budget']; h=mat.T@mat+np.diag(np.r_[np.full(k,ridge),np.full(z.shape[1],1e-8)]); c=mat.T@yy
 fit=minimize(lambda w:(.5*w@h@w-c@w,h@w-c), np.r_[np.full(k,bud/k),np.zeros(z.shape[1])],jac=True,bounds=[(0,cap)]*k+[(None,None)]*z.shape[1],constraints=[{'type':'eq','fun':lambda w:w[:k].sum()-bud,'jac':lambda w:np.r_[np.ones(k),np.zeros(z.shape[1])]}],method='SLSQP',options={'ftol':1e-13,'maxiter':400}); return fit.x, np.column_stack((S,z))@fit.x-y
if __name__=='__main__':
 rmse=lambda r:np.sqrt(np.mean(r*r)); print('y rms',rmse(y),'shifts',s['mu_shifts'],'n_unique_sensor',len(np.unique(inds)))
 for a,b in itertools.product('OCW',repeat=2):
  S=response(axis('x',a),axis('y',b)); w,r=fit(S,np.ones(len(y),bool)); cv=np.zeros_like(y)
  for j in np.unique(inds):
   tr=ids!=j; cv[~tr]=fit(S,tr)[1][~tr]
  print(a+b, 'full',round(rmse(r),7),'LOSO',round(rmse(cv),7),'worst',round(max(rmse(cv[ids==j]) for j in np.unique(ids)),7),'shift',np.round([rmse(i) for i in np.split(cv,len(s['mu_shifts']))],7),'weights',np.round(w[:k],3))
