import pickle
import numpy as np
from itertools import combinations_with_replacement
from scipy.optimize import least_squares
from scipy.linalg import svd
np.set_printoptions(precision=4,suppress=True,linewidth=150)
P='/holdout_task/'
s=pickle.load(open(P+'public_batch.pkl','rb'))[0]['study']
def load(path):
 ns={}; exec(compile(open(P+path).read(),path,'exec'),ns); return ns['run_node']
a=load('_frozen_nodes/node_00.py')({'__input__':s}); b=load('_frozen_nodes/node_01.py')({'node_00':a})
n=len(s['train']['x']); A=a['calibrated_x'][:n]; B=b['x'][:n]
u=s['panel_c']['u']; v=s['panel_c']['v1']; w=s['panel_c']['v2']
i,j=np.triu_indices(4,1)
def quad(x): return np.c_[np.ones(len(x)),x*x,x[:,i]*x[:,j]]
q=np.linalg.lstsq(quad(u),v,rcond=None)[0]; C=quad(B)@q
y=s['train']['y']; t=s['train']['traj']; F=np.c_[y,t]
# Use the four nonzero cyclic interactions identified independently on the panel.
def cyc(x): return np.c_[np.ones(len(x)),x*np.roll(x,-1,axis=1)]
C=cyc(B)@np.linalg.lstsq(cyc(u),v,rcond=None)[0]
def design(a,c): return np.c_[np.ones(len(a)),a,c,(a[:,:,None]*c[:,None,:]).reshape(len(a),-1)]
D=design(A,C); beta=np.linalg.lstsq(D,F,rcond=None)[0]; r=F-D@beta
print('fit RMSE',np.sqrt(np.mean(r*r,0))); print('largest y residuals',np.c_[y,r[:,0]][np.argsort(abs(r[:,0]))[-8:]])
for ix in [0,1,8]:
 print('channel',ix,'offset/linear',beta[:9,ix]); m=beta[9:,ix].reshape(4,4); print('cross weights',m,'sv',np.linalg.svd(m,compute_uv=False))
print('cross magnitudes over all horizons',np.linalg.norm(beta[9:,1:],axis=1).reshape(4,4))
print('input associations with residuals')
for name,z in [('A',A),('B',B),('C',C),('A^2',A*A),('C^2',C*C)]: print(name,np.corrcoef(np.c_[z,r[:,0],r[:,-1]].T)[:4,4:])
folds=[np.arange(52,72)]+list(np.array_split(np.random.default_rng(9).permutation(n),5))+[np.argsort(np.linalg.norm(A,axis=1))[-14:],np.argsort(np.linalg.norm(C,axis=1))[-14:]]
for method in ['full','ridge .002','ridge .02','ridge .10','sparse incremental','project y cubic','project y sine','project y asinh','state rank2','state rank3']:
 errs=[]
 for te in folds:
  tr=np.setdiff1d(np.arange(n),te); q=D[tr]; g=F[tr]; pred=D[te]@np.linalg.lstsq(q,g,rcond=None)[0]
  if method.startswith('ridge'):
   lam=float(method.split()[1]); p=np.sqrt(lam)*np.eye(q.shape[1]); p[0,0]=0; pred=D[te]@np.linalg.lstsq(np.r_[q,p],np.r_[g,np.zeros((len(p),g.shape[1]))],rcond=None)[0]
  if method=='sparse incremental':
   for k in range(2):
    z=g[:,0] if k==0 else g[:,1:]; keep=[0]
    while True:
     err,feature=min((np.linalg.norm(z-q[:,keep+[j]]@np.linalg.lstsq(q[:,keep+[j]],z,rcond=None)[0]),j) for j in range(1,D.shape[1]) if j not in keep); keep.append(feature)
     if len(keep)==(10 if k==0 else 12): break
    fit=D[te][:,keep]@np.linalg.lstsq(q[:,keep],z,rcond=None)[0]
    if k==0: pred[:,0]=fit
    else: pred[:,1:]=fit
  if method.startswith('project'):
   # Compare one-parameter monotonic transforms, profiled within each training fold.
   from scipy.optimize import minimize_scalar
   if method.endswith('cubic'):
    f=lambda z,k:z+k*z**3
    inv=lambda x,k: np.array([np.polynomial.polynomial.polyroots([-xx,1,0,k])[np.argmin(abs(np.polynomial.polynomial.polyroots([-xx,1,0,k]).imag))].real if k>1e-9 else xx for xx in x])
   elif method.endswith('sine'):
    f=lambda z,k:np.arcsin(np.clip(k*z,-.999,.999))/k; inv=lambda z,k:np.sin(k*z)/k
   else:
    f=lambda z,k:np.arcsinh(k*z)/k; inv=lambda z,k:np.sinh(k*z)/k
   upper=max(.0002,.99/max(abs(g[:,0]))) if method.endswith('sine') else 4
   def get(k):
    lag=f(g[:,0],k); b=np.linalg.lstsq(q,lag,rcond=None)[0]; return inv(q@b,k),inv(D[te]@b,k)
   k=minimize_scalar(lambda k:np.mean((get(k)[0]-g[:,0])**2),bounds=(.0001,upper),method='bounded').x; pred[:,0]=get(k)[1]
  if method.startswith('state'):
   rank=int(method[-1]); basis=np.linalg.svd(g[:,1:],full_matrices=False)[2][:rank]; pred[:,1:]=pred[:,1:]@basis.T@basis
  er=np.abs(pred-F[te]); errs.append([er[:,0].mean(),er[:,1:].mean()])
 print(method,'official',errs[0],'random worst',np.max(errs[1:6],axis=0),'tails',np.array(errs[6:]))
