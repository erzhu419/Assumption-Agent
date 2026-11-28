import numpy as np,pickle,sys
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
from scipy.linalg import solve
from itertools import combinations_with_replacement
sys.path.insert(0,'/tmp/hegel_continual_archive')
from conf_m102_a04_batch_00_solver import _align, _panel_maps
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; A=s['panel_a']; t=s['eval_times']; T=tr['traj']; Y=np.c_[tr['y'],T]; n=len(Y)
a,b=_align(A); z=(tr['x']-a)/b
np.set_printoptions(precision=5,suppress=True)
print('align',a,b,'resid quantiles',np.quantile(abs(A['v']-a-A['u']*b),[.2,.5,.7,.8,.9,.95,1],axis=0),'z ranges',z.min(0),z.max(0)); print('sv centered',np.linalg.svd(T-T.mean(0),compute_uv=False))
for p in ['panel_b','panel_c']:
 d=s[p]; u=d['u'];
 for k in d:
  if k=='u':continue
  v=d[k]; print(p,k,'corr',np.corrcoef(u.T,v.T)[:4,4:]);
  for deg in [1,2,3]:
   F=np.column_stack([np.ones(len(u))]+[np.prod(u[:,j],axis=1) for m in range(1,deg+1) for j in combinations_with_replacement(range(4),m)]); print('poly',deg,np.sqrt(np.mean((F@np.linalg.lstsq(F,v,rcond=None)[0]-v)**2,axis=0)))
for name,F in [('exp2',lambda r:-np.expm1(-t[:,None]*r)),('exp2lin',lambda r:np.c_[t,-np.expm1(-t[:,None]*r)]),('exp3',lambda r:-np.expm1(-t[:,None]*r)),('exp2const',lambda r:np.c_[np.ones(len(t)),-np.expm1(-t[:,None]*r)]),('expsin',lambda r:np.c_[-np.expm1(-t[:,None]*r[:2]),np.sin(t*r[2])])]:
 def fun(r):
  f=F(r);return (f@np.linalg.lstsq(f,T.T,rcond=None)[0]-T.T).ravel()
 r=least_squares(fun,[.4,1.8,1.1] if name in ['exp3','expsin'] else [.4,1.8],bounds=(.01,10)).x;print(name,r,np.linalg.norm(fun(r))/np.sqrt(T.size))
folds=[(np.arange(52),np.arange(52,72))]+[(p[:52],p[52:]) for p in [np.random.default_rng(i).permutation(n) for i in range(5)]]
def cv(K,l):
 er=[]
 for a,b in folds:
  P=K[np.ix_(b,a)]@solve(K[np.ix_(a,a)]+l*np.eye(len(a)),Y[a],assume_a='pos');d=abs(P-Y[b]);er.append([d[:,0].mean(),d[:,1:].mean()])
 return np.array(er)
for name,X in [('z',z),('tanh',np.tanh(z)),('soft',z/(1+abs(z))),('z+soft',np.c_[z,z/(1+abs(z))]),('archC',_panel_maps(s['panel_c'],z))]:
 X=(X-X.mean(0))/np.maximum(.06,X.std(0)); G=X@X.T/X.shape[1]; D=cdist(X,X,'sqeuclidean')/X.shape[1]; out=[]
 for p in [1,2,3,4]:
  for amp in [0,.1,1]:
   for sc in ([.25,1,4] if amp else [1]):
    K=(1+.3*G)**p+amp*np.exp(-sc*D)
    for l in [1e-6,1e-4,.002,.02]:
     e=cv(K,l);out.append((e[1:].mean(),p,amp,sc,l,e[0],e[1:].max(0)))
 print(name, sorted(out,key=lambda row:row[0])[:3])
