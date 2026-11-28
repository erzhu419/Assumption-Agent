import numpy as np,pickle,itertools
from scipy.linalg import solve
from scipy.spatial.distance import cdist
from adaptive import coords
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];x=s['train']['x']; y=s['train']['y']; T=s['train']['traj']; Z=coords(s,x); n=len(y)
p=s['panel_c']; u=p['u']; print('times telemetry summary',s['eval_times'],s['telemetry_vec'],s['train']['summary']);
for f in [np.arcsinh, np.arctan, np.tanh, lambda x:x/np.sqrt(1+x*x),lambda x:x/(1+abs(x))]:
 from scipy.optimize import minimize_scalar
 def fit(c):
  H=np.c_[np.ones(len(u)),f(c*u)];w=np.linalg.lstsq(H,p['v1'],rcond=None)[0];return np.std(H@w-p['v1']),w
 q=minimize_scalar(lambda h:fit(h)[0],bounds=(.01,8),method='bounded'); print(f,q.x,q.fun,np.round(fit(q.x)[1],3))
print('V2 coef',np.round(np.linalg.lstsq(np.c_[np.ones(len(u)),u,u*u/(1+.1625*u*u)],p['v2'],rcond=None)[0],3))
folds=[(np.arange(52),np.arange(52,n))]; rng=np.random.default_rng(821)
for _ in range(8):
 i=rng.permutation(n);folds.append((i[:52],i[52:]))
def score(K, r):
 e=[np.mean(abs(K[np.ix_(j,i)]@solve(K[np.ix_(i,i)]+r*np.eye(len(i)),y[i],assume_a='pos')-y[j])) for i,j in folds]; return np.mean(e),e[0],max(e)
results=[]
for k,z in enumerate(Z):
 d=z.shape[1]; l=z@z.T; D=cdist(z,z,'sqeuclidean'); pair=np.array([v for v in itertools.product([0,-1,1],repeat=d) if 0<sum(q!=0 for q in v)<=d and next(q for q in v if q)!=-1]); pair=pair/np.linalg.norm(pair,axis=1)[:,None];
 for famil in ['poly','rat_rad','rat_sep','rat_l1','rat_proj','trig_proj','trig_sep','asinh','atan','tanh','poly_diag']:
  for a in [.04,.11,.22,.38,.65,1.,1.25,1.6,2.]:
   H=None; kernel=None
   if famil in ['rat_proj','trig_proj']:
    q=z@pair.T
    H=np.c_[q/(1+a*q*q),1/(1+a*q*q)] if famil=='rat_proj' else np.c_[np.sin(a*q),np.cos(a*q)]; H/=np.sqrt(len(pair))
   if famil=='trig_sep': H=np.c_[np.sin(a*z),np.cos(a*z)];
   if famil in ['asinh','atan','tanh']:
    q={'atan':np.arctan,'asinh':np.arcsinh,'tanh':np.tanh}[famil](a*z)/a;kernel=(1+q@q.T)**3
   if famil.startswith('rat_') and H is None:
    q={'rat_rad':(z*z).sum(1),'rat_sep':(z*z).max(1),'rat_l1':(abs(z).sum(1))**2}[famil]; f=1/(1+a*q);kernel=(1+l)**3*f[:,None]*f[None,:]
   if famil=='poly': kernel=(1+a*l)**4
   if famil=='poly_diag': kernel=(1+a*l)**3+2*np.sum((1+a*z[:,None,:]*z[None,:,:])**5,axis=-1)
   if H is not None: kernel=H@H.T
   for g in [0,.035,.25]:
    for h in [.28,.52]:
     C=1+l+.12*l*l+kernel+g*np.exp(-h*D)
     for r in [1e-5,8e-5,.0005,.002]:results.append((*score(C,r),k,famil,a,g,h,r))
  for g in [.15,.5,2]:
   for a in [.07,.11,.15,.19,.23,.3,.38,.48,.6,.75]:
    if famil!='poly':continue
    for p in [0,1,2,3]:
     for r in [1e-6,1e-5,6e-5,.0002,.001]:
      results.append((*score((1+l*.27)**p+g*np.exp(-a*D),r),k,'rbf',a,g,p,r))
print('mean',*sorted(results)[:14],sep='\n');print('gate',*sorted(results,key=lambda a:a[1])[:14],sep='\n');print('worst',*sorted(results,key=lambda a:a[2])[:8],sep='\n')
