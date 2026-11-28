import pickle,runpy,itertools
import numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; old=runpy.run_path('/tmp/hegel_continual_archive/dev_m101_c04_batch_00_solver.py')
np.set_printoptions(precision=4,suppress=True,linewidth=140)
def ls(X,Y):return np.linalg.lstsq(X,Y,rcond=None)[0]
def aff(X):return np.c_[np.ones(len(X)),X]
def ga(x):return np.exp(-x*x)-1
def rat(x):return x/(1+x*x)
def hinge(x):return np.maximum(x,0)
cal=old['_calibration'](s); x=s['train']['x']; q=(x-cal[0])/cal[1]; z=old['_coordinates'](x,cal); c=old['_cross'](z); y=s['train']['y']; C=s['panel_c']; u=C['u']
# The second output is nearly affine plus a rational square; profile each channel's shape independently.
for key, families in [('v1',{'tanh':np.tanh,'sin':np.sin,'rat':rat,'atan':np.arctan,'erf':__import__('scipy').special.erf}),('v2',{'evenrat':lambda t:t*t/(1+t*t),'cos':np.cos,'gauss':ga})]:
 for name,f in families.items():
  for shift in (False,True):
   def panel(p,u):return aff(u) if len(p)==0 else np.c_[aff(u),f(u*np.exp(p[:4])+ (p[4:] if shift else 0))]
   def err(p):
    X=panel(p,u);return (X@ls(X,C[key])-C[key]).ravel()
   n=8 if shift else 4; lo=np.r_[np.repeat(-1.8,4),np.repeat(-1.,n-4)]; hi=np.r_[np.repeat(1.7,4),np.repeat(1.,n-4)]
   fits=[least_squares(err,np.r_[np.full(4,k),np.zeros(n-4)],bounds=(lo,hi),max_nfev=120) for k in (-.4,.4,1.1)]; r=min(fits,key=lambda r:r.cost)
   loo=[]
   for i in range(len(u)):
    idx=np.arange(len(u))!=i
    def e(p):
     X=panel(p,u[idx]);return (X@ls(X,C[key][idx])-C[key][idx]).ravel()
    f0=least_squares(e,r.x,bounds=(lo,hi),max_nfev=60); X=panel(f0.x,u);loo.extend((X[i]@ls(X[idx],C[key][idx])-C[key][i]).tolist())
   print(key,name,shift,'fit/LOO',np.sqrt(np.mean(err(r.x)**2)),np.sqrt(np.mean(np.square(loo))), 'scale',np.exp(r.x[:4]),'shift',r.x[4:])
# Fit scalar ridge features by variable projection; restart and select on fitted rows only.
rng=np.random.default_rng(125); folds=[(np.arange(56),np.arange(56,72))]+[(p[:56],p[56:]) for p in [rng.permutation(72) for _ in range(4)]]
starts=np.r_[np.eye(4),-np.eye(4),[np.eye(4)[i]+sg*np.eye(4)[j] for i,j in itertools.combinations(range(4),2) for sg in (1,-1)],rng.normal(size=(8,4))]
for label,B in [('cross',np.c_[aff(z),c]),('quad',np.c_[aff(z),*[z[:,i]*z[:,j] for i,j in itertools.combinations_with_replacement(range(4),2)]])]:
 for name,f in [('sin',np.sin),('tanh',np.tanh),('rational',rat),('cos',np.cos),('gauss',ga),('hinge',hinge)]:
  scores=[]; details=[]
  for a,b in folds:
   def des(p,I):return np.c_[B[I], f(z[I]@p[:4]+p[4])]
   def fit(p):
    X=des(p,a);return X@ls(X,y[a])-y[a]
   init=[np.r_[v*k,d] for v in starts for k,d in ((1.7,0),(3.,.8))]; sol=min((least_squares(fit,p,bounds=(-10,10),max_nfev=90) for p in init),key=lambda r:r.cost); w=ls(des(sol.x,a),y[a]);scores.append(np.mean(abs(des(sol.x,b)@w-y[b])));details.append((np.sqrt(2*sol.cost/len(a)),sol.x,w[-1]));
  print(label,name,'CV',np.round(scores,5),'train rmse, p, amplitude',details[:2],flush=True)
