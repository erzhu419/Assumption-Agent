import sys,itertools,pickle,numpy as np
sys.path.insert(0,'/holdout_task');from batch_solver import _operator,_ls,_aff,_scalar,_state
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];d=s['train'];w=_operator(s,d['x']); y=d['y']; T=d['traj']; n=52
norm=np.std(T,0); P=list(itertools.combinations(range(4),2))
def products(z): return np.column_stack([z[:,i]*z[:,j] for i,j in P])
def fn(k,z):
 if k=='atan':return np.arctan(z)
 if k=='tanh':return np.tanh(z)
 if k=='rat':return z/(1+z*z)
 if k=='sin':return np.sin(z)
 if k=='sqrt':return z/np.sqrt(1+z*z)
 if k=='asinh':return np.arcsinh(z)
 if k=='gauss':return np.exp(-z*z)
 if k=='quad':return z*z
 if k=='cube':return z*z*z
res=[]
for typ in ('atan','tanh','rat','sin','sqrt','asinh'):
 for targ in ('scalar','state'):
  Y=y if targ=='scalar' else T
  for mode in ('add','plain','rawpair','saturpair','mixpair'):
   for npar in (1,4,8):
    def features(p):
     k = p[0] if npar==1 else p[:4]; shift=p[4:] if npar==8 else 0.; Z=fn(typ,w*k+shift)
     if mode=='plain':return _aff(Z)
     X=np.c_[_aff(w),Z]
     if mode=='add':return X
     if mode=='rawpair':return np.c_[X,w*w,products(w)]
     if mode=='saturpair':return np.c_[X,Z*Z,products(Z)]
     return np.c_[X,*[w[:,i]*Z[:,j]+w[:,j]*Z[:,i] for i,j in P],w*Z]
    def err(p):
     F=features(p)[:n]; R=F@_ls(F,Y[:n])-Y[:n];return (R if targ=='scalar' else R/norm).ravel()
    b=(np.r_[np.repeat(.25,min(4,npar)),np.repeat(-1.,max(0,npar-4))],np.r_[np.repeat(5.,min(4,npar)),np.repeat(1.,max(0,npar-4))]);q=least_squares(err,np.r_[np.repeat(1.2,min(4,npar)),np.zeros(max(0,npar-4))],bounds=b,max_nfev=120);F=features(q.x);r=F@_ls(F[:n],Y[:n])-Y
    res.append((targ,np.mean(abs(r[n:])),np.mean(abs(r[:n])),typ,mode,npar,np.round(q.x,4)))
for targ in ('scalar','state'):
 rows=[row[1:] for row in res if row[0]==targ]
 for k in [0,1]:
  print(targ,k);print(*sorted(rows,key=lambda r:r[k])[:7],sep='\n')
print('state centered temporal singular values',np.linalg.svd(T-T.mean(0),compute_uv=False));print('uncentered',np.linalg.svd(T,compute_uv=False));
for name,F,Y in [('scalar',_scalar(w),y),('state',_state(w),T)]:
 r=F@_ls(F[:n],Y[:n])-Y; print(name,'baseline',np.mean(abs(r[n:])), 'high input radius',np.mean(abs(r[n:][np.linalg.norm(w[n:],axis=1)>np.median(np.linalg.norm(w[n:],axis=1))])))
