import sys,pickle,numpy as np
from scipy.optimize import least_squares
from scipy import sparse
from scipy.sparse.linalg import splu
sys.path.insert(0,'/holdout_task'); import batch_solver as m
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; nx,ny=s['nx'],s['ny']; y=m._calibrate(s); Z=s['Z_nuisance']; at=s['sensor_ij']@np.array([1,nx]); rhs=s['source_modes'].transpose(0,2,1).reshape(-1,nx*ny).T; I=sparse.eye(nx*ny,format='csc'); w0=np.full(rhs.shape[1],s['total_budget']/rhs.shape[1]); Q=np.linalg.qr(np.r_[np.ones((1,len(w0))),np.eye(len(w0))][1:,:] + 0.)[0]; Q=np.linalg.svd(np.ones((1,len(w0))),full_matrices=True)[2][1:].T
L0=m._operator(s,('reflect','reflect')); mats=[]
for typ in ['leak','conductance','wrap']:
 for direction,n in [('x',nx),('y',ny)]:
  h=s['L'+direction]/(n-1); k=s['kappa_'+direction]
  for j,foo,bar in [(0,1,n-1),(n-1,n-2,0)]:
   A=np.zeros((n,n)); A[j,j]=1
   if typ=='conductance': A[j,foo]=-1; A*=k[j]/h**2
   elif typ=='wrap': A[j,bar]=-1; A*=.5*(k[j]+k[bar])/h**2
   else: A*=s['robin_'+direction]/h
   mats.append(sparse.kron(sparse.eye(ny),A,format='csc') if direction=='x' else sparse.kron(A,sparse.eye(nx),format='csc'))
def design(p):
 L=L0+sum((p[i]*a for i,a in enumerate(mats[:len(p)])),start=sparse.csc_matrix(L0.shape)); S=np.vstack([splu(L+mu*I).solve(rhs)[at] for mu in s['mu_shifts']]); return S,np.c_[S@Q,Z]
def fit(p,train):
 S,D=design(p); b=y-S@w0; coef=np.linalg.lstsq(D[train],b[train],rcond=None)[0]; return D@coef-b,w0+Q@coef[:Q.shape[1]]
all_=np.ones(len(y),bool); folds=[(np.arange(len(y))%len(at))//4%4,np.arange(len(y))//len(at)]
for count in (0,4,8,12):
 def opt(mask):
  if not count:return np.empty(0)
  floor=np.full(count,-.8); ceil=np.full(count,2.); floor[:4]=-.2
  return least_squares(lambda x:fit(x,mask)[0][mask],np.zeros(count),bounds=(floor,ceil),max_nfev=120).x
 p=opt(all_); err,w=fit(p,all_); scores=[]
 for f in folds:
  scores.append([round(float(np.sqrt(np.mean(fit(opt(f!=j),f!=j)[0][f==j]**2))),6) for j in np.unique(f)])
 print('boundary_parameters', count, 'rmse',np.sqrt(np.mean(err**2)), 'parameters',p.round(4),'w',w.round(4),'heldout',scores)
