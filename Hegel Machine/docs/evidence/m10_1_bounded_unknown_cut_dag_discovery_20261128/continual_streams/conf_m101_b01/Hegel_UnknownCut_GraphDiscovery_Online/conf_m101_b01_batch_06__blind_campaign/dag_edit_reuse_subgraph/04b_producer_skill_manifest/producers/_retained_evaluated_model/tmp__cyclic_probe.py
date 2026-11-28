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
from scipy.optimize import minimize_scalar
P=np.roll(A,-1,axis=1)*C
one=np.ones(len(A)); folds=[np.arange(52,72)]+list(np.array_split(np.random.default_rng(9).permutation(n),5))+[np.argsort(np.linalg.norm(A,axis=1))[-14:],np.argsort(np.linalg.norm(C,axis=1))[-14:]]
for label,X in [('C+sum',np.c_[C,P.sum(1)]),('1+C+sum',np.c_[one,C,P.sum(1)]),('1+A0+C+sum',np.c_[one,A[:,0],C,P.sum(1)]),('1+A+C+sum',np.c_[one,A,C,P.sum(1)]),('1+C+cyclic',np.c_[one,C,P]),('1+A+C+cyclic',np.c_[one,A,C,P]),('1+C+sum+A2',np.c_[one,C,P.sum(1),A*A])]:
 b=np.linalg.lstsq(X,F,rcond=None)[0]; r=F-X@b; print(label,'y coefficient',b[:,0],'trajectory last',b[:,-1],'train RMSE',np.sqrt(np.mean(r*r,0))[[0,1,-1]])
 for mode in ['linear','tanh','sin','cubic']:
  er=[]; ks=[]
  for te in folds:
   tr=np.setdiff1d(np.arange(n),te); x=X[tr]; z=y[tr]; bt=np.linalg.lstsq(x,F[tr],rcond=None)[0]; pred=X[te]@bt
   if mode!='linear':
    def link(g,k):
     if mode=='tanh': return np.tanh(k*g)/k
     if mode=='sin': return np.sin(k*g)/k
     return g-k*g**3
    def residual(p): return link(x@p[:-1],p[-1])-z
    fit=least_squares(residual,np.r_[bt[:,0],.3 if mode!='cubic' else .05],bounds=(np.r_[np.full(x.shape[1],-np.inf),1e-7],np.r_[np.full(x.shape[1],np.inf),1.5 if mode!='cubic' else .4]),max_nfev=75).x
    ks.append(fit[-1]); pred[:,0]=link(X[te]@fit[:-1],fit[-1])
   e=abs(pred-F[te]); er.append([e[:,0].mean(),e[:,1:].mean()])
  print(mode,'official',np.round(er[0],5),'random worst',np.max(er[1:6],axis=0),'tails',np.array(er[6:]),'link',np.round(ks,3))
X=np.c_[one,C,P.sum(1)]; b=np.linalg.lstsq(X,F,rcond=None)[0]; print('time coeff',b[:,1:]); r=y-X@b[:,0]; print('residual vs standard proposal'); print(np.corrcoef(np.c_[r,A,A*A,B,np.cos(B),C*C,(C*C).sum(1), (X@b[:,0])**3].T)[0,1:]); print('sorted residual',np.c_[X@b[:,0],r][np.argsort(X@b[:,0])][::4])
