import numpy as np,pickle,itertools,glob,runpy
from scipy.optimize import least_squares
from scipy.special import expit
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; a=runpy.run_path(glob.glob('/tmp/hegel_continual_archive/*01_solver.py')[0]); c=a['_calibration'](s['panel_a']); z=a['_inverse'](s['train']['x'],c); y=s['train']['y']; Y=s['train']['traj']; t=s['eval_times'];
np.set_printoptions(precision=5,suppress=True,linewidth=140)
def fit(A,y):return np.linalg.lstsq(A,y,rcond=1e-10)[0]
def poly(z,d):
 return np.column_stack([np.ones(len(z))]+[np.prod(z[:,ix],axis=1) for k in range(1,d+1) for ix in itertools.combinations_with_replacement(range(z.shape[1]),k)])
def cv(A,Y):
 P=A@np.linalg.pinv(A); r=A@fit(A,Y)-Y; return np.mean(abs(r/(1-np.diag(P)).reshape((-1,)+(1,)*(Y.ndim-1)))),np.sqrt(np.mean(r*r))
print('calibration',c,'latent range',z.min(0),z.max(0));
for d in (1,2,3,4):print('poly',d,cv(poly(z,d),y),cv(poly(z,d),Y))
for p,key in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 u,v=s[p]['u'],s[p][key]; L=poly(u,1); Q=poly(u,2); print('\n',p,key,'ranges',v.min(0),v.max(0)); print('cv linear/quadratic',cv(L,v),cv(Q,v)); print('quadratic',fit(Q,v).T)
 for fun in (np.sin,np.tanh,lambda x:x/(1+x*x),lambda x: np.logaddexp(x,0),lambda x:x*x):
  fits=[]
  for j in range(4):
   w=v[:,j]; h=fit(L,w); best=None
   for ini in (h,h*.5,h*2,np.r_[0,h[1:]/np.linalg.norm(h[1:])]):
    for base in (np.ones((len(u),1)),L):
     def residual(h):
      M=np.column_stack((base,fun(L@h)));return M@fit(M,w)-w
     r=least_squares(residual,ini,max_nfev=200); score=np.sqrt(np.mean(r.fun**2));
     if best is None or score<best[0]:best=(score,r.x,base.shape[1]);
   fits.append(best)
  print(getattr(fun,'__name__'),fits)
print('target patterns',fit(poly(z,1),y)); print('traj temporal modes',np.linalg.svd(Y,full_matrices=False)[2][:4]); print('rank-one fit to degree two residual by time',np.std(Y-poly(z,2)@fit(poly(z,2),Y),axis=0))
