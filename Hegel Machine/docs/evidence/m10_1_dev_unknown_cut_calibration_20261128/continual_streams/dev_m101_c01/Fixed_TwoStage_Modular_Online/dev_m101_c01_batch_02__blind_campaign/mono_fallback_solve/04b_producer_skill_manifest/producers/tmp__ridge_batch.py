import numpy as np,pickle,itertools as it
from scipy.optimize import least_squares
from scipy.linalg import solve
from scipy.spatial.distance import cdist

def poly(x,d):
 return np.column_stack([np.prod(x[:,j],axis=1) for k in range(d+1) for j in it.combinations_with_replacement(range(x.shape[1]),k)])

def fit(x,y,q,kind,d=2,starts=12):
 P=poly(x,d); T=poly(q,d); X=poly(x,1); W=poly(q,1); rng=np.random.default_rng(123)
 U,S,V=np.linalg.svd(P,full_matrices=False); Q=U[:,:P.shape[1]]; R=(V.T/S)@U.T
 if kind=='quot':
  fun=lambda z: z/np.sqrt(1+z*z)
 elif kind=='soft': fun=np.tanh
 elif kind=='sin': fun=np.sin
 elif kind=='gauss': fun=lambda z:np.exp(-z*z)
 elif kind=='ratplus': fun=lambda z:1/(1+z*z)
 elif kind=='pow': fun=lambda z:z**3
 def design(p,H,L):
  z=L@p[:5]
  if kind=='rat':return H/(1+z[:,None]**2)
  if kind=='ratmix':return np.c_[L,H/(1+z[:,None]**2)]
  if kind=='ratbias':return H/(1+(z[:,None]+np.sum(L[:,1:]**2,axis=1,keepdims=True)*p[5])**2)
  if kind=='link':return np.c_[H,np.tanh(z),np.cos(z)]
  if kind=='double':return np.c_[H,np.tanh(z),np.tanh(L@p[5:])]
  if kind=='tmod':return np.c_[H,L*np.tanh(z[:,None])]
  return np.c_[H,fun(z)]
 def res(p):
  H=design(p,P,X); c=np.linalg.lstsq(H,y,rcond=1e-10)[0];return np.r_[(H@c-y).ravel(),0.0002*p]
 # Start from diverse orientations; rank starts by initial training residual.
 k=10 if kind=='double' else 6 if kind=='ratbias' else 5
 init=[rng.normal(0,scale,k) for scale in (.3,.75,1.5) for _ in range(starts//3)]; init+=[np.r_[0.,j,np.zeros(k-5)] for j in np.eye(4)]
 sol=min((least_squares(res,p,max_nfev=95,ftol=1e-7,gtol=1e-7) for p in init),key=lambda v:v.cost)
 H=design(sol.x,P,X); c=np.linalg.lstsq(H,y,rcond=1e-10)[0];return design(sol.x,T,W)@c,np.sqrt(np.mean((H@c-y)**2)),sol.x

if __name__=='__main__':
 s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; y=tr['y']; z=tr['traj']; t=s['eval_times']; print('ranges',x.min(0),x.max(0));print('sv',np.linalg.svd(z,compute_uv=False)); print('summary',tr['summary'],np.array([y.mean(),y.std(),z[:,-1].mean(),z[:,-1].std()]))
 n=len(x); folds=[np.arange(56,72),*np.array_split(np.random.default_rng(67).permutation(n),4)]
 for k in ('rat','ratmix','ratbias','soft','sin','link','gauss','quot','ratplus','double','tmod'):
  rr=[]; pp=[]
  for j in folds:
   i=np.setdiff1d(np.arange(n),j); p,a,b=fit(x[i],y[i],x[j],k,starts=9);rr.append([np.mean(abs(p-y[j])),a]);pp.append(b)
  print(k,np.round(rr,5).T,'p0',np.round(pp[0],3),flush=True)
 # Fit each trajectory mode with the strongest compact grammars as a separate test.
 for k in ('rat','ratmix','soft','sin','link','double','tmod'):
  rr=[]
  for j in folds:
   i=np.setdiff1d(np.arange(n),j); V=np.linalg.svd(z[i],full_matrices=False)[2][:2]; amps=z[i]@V.T
   p=[fit(x[i],a,x[j],k,starts=6)[0] for a in amps.T]; rr.append(np.mean(abs(np.array(p).T@V-z[j])))
  print('traj',k,np.round(rr,5),flush=True)
