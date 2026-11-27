import sys
sys.path.insert(0,'/holdout_task')
import itertools,pickle,numpy as np
from scipy.linalg import null_space
from scipy.sparse import diags,kron,eye
from scipy.sparse.linalg import splu
from _verified_upstream import calibrate_upstream
s=calibrate_upstream(pickle.load(open('public_batch.pkl','rb'))[0]['study'])
nx,ny=s['nx'],s['ny']; n=nx*ny
modes=s['source_modes']; B=np.column_stack([x.ravel(order='F') for x in modes]); m=B.shape[1]
ix=s['sensor_ij']@np.array([1,nx]); z=s['Z_nuisance']; y=s['y_obs_raw']; shifts=s['mu_shifts']; ident=eye(n,format='csc'); budget=s['total_budget']; w0=np.full(m,budget/m); N=null_space(np.ones((1,m)))
def axis(d,kind):
 k=s['kappa_'+d]; v=s['vel_'+d]; h=s['L'+d]/(len(k)-1)
 west=(k+np.roll(k,1))/(2*h*h)+np.maximum(v,0)/h; east=(k+np.roll(k,-1))/(2*h*h)+np.maximum(-v,0)/h
 A=np.diag(west+east)+np.diag(-west[1:],-1)+np.diag(-east[:-1],1)
 if kind=='P': A[0,-1]=-west[0]; A[-1,0]=-east[-1]
 else:
  leak=s['robin_'+d]/h if kind=='O' else 0
  A[0,0]=k[0]/h**2+leak; A[-1,-1]=k[-1]/h**2+leak; A[0,1]=-k[0]/h**2; A[-1,-2]=-k[-1]/h**2
 return A
def forward(ax,ay):
 L=(kron(eye(ny),ax)+kron(ay,eye(nx))+diags(s['V_reaction_2d'].ravel(order='F'))).tocsc()
 return np.vstack([splu(L+mu*ident).solve(B)[ix] for mu in shifts])
def fit(S,t):
 X=np.column_stack((S@N,z)); offset=S@w0
 c=np.linalg.lstsq(X[t],(y-offset)[t],rcond=None)[0]
 return offset+X@c,w0+N@c[:m-1]
groups=np.tile(np.arange(len(ix))//4,len(shifts)); scores={}
for a,b in itertools.product('ORP',repeat=2):
 S=forward(axis('x',a),axis('y',b)); pred,w=fit(S,np.ones(len(y),bool)); r=pred-y
 folds=np.array([np.sqrt(np.mean((fit(S,groups!=g)[0][groups==g]-y[groups==g])**2)) for g in np.unique(groups)])
 scores[a+b]=folds
 print(a+b,'RMSE full/OOF/worst',*[round(x,7) for x in (np.sqrt(np.mean(r*r)),np.sqrt(np.mean(folds**2)),max(folds))],'shift RMSE',np.sqrt(np.mean(r.reshape(len(shifts),-1)**2,axis=1)).round(6),'weight extrema',w.min().round(4),w.max().round(4))
print('paired RR-vs-PR fold RMSE',np.column_stack((scores['RR'],scores['PR'])).round(7))
