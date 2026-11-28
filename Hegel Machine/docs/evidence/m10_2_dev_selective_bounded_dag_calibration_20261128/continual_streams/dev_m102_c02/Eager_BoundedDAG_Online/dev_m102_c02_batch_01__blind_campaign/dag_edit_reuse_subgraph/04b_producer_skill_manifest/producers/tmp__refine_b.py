exec(open('/tmp/panel_discovery.py').read().split('allmaps=[]')[0])
from scipy.optimize import least_squares
u,y=s['panel_b']['u'],s['panel_b']['v']; te=s['eval_times']; funs.update({'atan':lambda x,k:np.arctan(k*x),'softsign':lambda x,k:x/(1+k*abs(x)), 'erf':__import__('scipy.special',fromlist=['erf']).erf})
def make(u,f,k,lin):return np.column_stack([np.ones(len(u))]+([u] if lin else [])+[f(u,k)])
results=[]
for name in ('sin','tanh','rat1','rat2','atan','softsign','erf'):
 f=funs[name]
 for lin in (0,1):
  for fre in (0,1):
   def fit(U,Y):
    def res(k):
     A=make(U,f,k,lin); return (A@np.linalg.lstsq(A,Y,rcond=None)[0]-Y).ravel()
    grids=np.linspace(.2,5.2,19); starts=sorted(grids,key=lambda t:np.linalg.norm(res(np.array([t]))))[:3]; best=None
    for p in starts:
     k=np.full(4 if fre else 1,p); opt=least_squares(res,k,bounds=(.03,8.),max_nfev=180,ftol=1e-10); loss=np.linalg.norm(opt.fun)
     if best is None or loss<best[0]:best=loss,opt.x
    loss,k=best; A=make(U,f,k,lin); return k,np.linalg.lstsq(A,Y,rcond=None)[0],loss/(Y.size**.5)
   k,B,err=fit(u,y); loo=[]
   for i in range(len(u)):
    mask=np.arange(len(u))!=i; ki,Bi,_=fit(u[mask],y[mask]); loo.extend((make(u[i:i+1],f,ki,lin)@Bi-y[i]).ravel())
   c=make(x,f,k,lin)@B; A=np.c_[np.ones(len(c)),c]; D=np.c_[A,(c[:,:2]*c[:,2:]).sum(1)]; ix=52
   cv=tuple(np.mean(abs(F[ix:]@np.linalg.lstsq(F[:ix],z[:ix],rcond=None)[0]-z[ix:])) for F,z in [(D,tr['y']),(A,tr['traj'])]); results.append((np.mean(np.square(loo))**.5,name,lin,fre,err,k,cv));print(results[-1],flush=True)
print('trajectory time grid',te,'singular spectrum:',np.linalg.svd(tr['traj'],compute_uv=False)); print('rank basis result',sorted(results,key=lambda z:z[0])[:5]);
