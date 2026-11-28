import itertools as it,pickle,numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; p=s['panel_c']; u=p['u']; y=p['v2']; A=np.c_[np.ones(len(u)),u]; Q=np.linalg.qr(A)[0]; center=lambda z:z-Q@(Q.T@z); R=center(y); rng=np.random.default_rng(45)
results=[]
for j in (1,2,3):
 v=np.roll(u,-j,axis=1)
 families={'sin':np.sin,'tanh':np.tanh,'atan':np.arctan,'rat':lambda z:z/(1+z*z),'cosh':lambda z:np.cosh(np.clip(z,-20,20)),'softplus':lambda z:np.logaddexp(0,z)}
 for name,fn in families.items():
  for mode in ('phase','phase_plus_quad','ratio_mul','ratio_cross') if name=='rat' else ('phase','phase_plus_quad'):
   def design(k):
    if mode=='ratio_mul': return (u*v+k[0]*u*u+k[1]*v*v)/(1+(k[2]*u+k[3]*v)**2)
    if mode=='ratio_cross': return (u*v+k[0]*u+k[1]*v+k[2])/(1+(k[3]*u+k[4]*v+k[5])**2)
    z=k[0]*u+k[1]*v+k[2]*u*v+k[3]
    if mode=='phase_plus_quad':z=z+k[4]*u*u+k[5]*v*v
    return fn(z)
   n=4 if mode in ('phase','ratio_mul') else 6
   def objective(k):
    B=center(design(k)); return (B@np.linalg.lstsq(B,R,rcond=1e-9)[0]-R).ravel()
   seeds=[rng.normal(0,1,n) for i in range(14)]; seeds += [np.array([-.12,1.05,.25,.45]+[0]*(n-4)),np.array([1.05,-.12,.25,.45]+[0]*(n-4))]
   sol=min((least_squares(objective,k,bounds=(-7,7),max_nfev=150,ftol=1e-9) for k in seeds),key=lambda z:z.cost); B=np.c_[A,design(sol.x)]; qq=np.linalg.qr(B)[0]; r=objective(sol.x).reshape(y.shape); press=np.sqrt(np.mean((r/np.maximum(1-(qq*qq).sum(1)[:,None],1e-10))**2)); results.append((np.sqrt(np.mean(r*r)),press,j,name,mode,sol.x.tolist()))
print(*sorted(results)[:20],sep='\n')
