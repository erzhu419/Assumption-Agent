import sys,pickle,numpy as np
from scipy.optimize import least_squares
from scipy.linalg import qr
from itertools import combinations_with_replacement
sys.path.insert(0,'/tmp/hegel_continual_archive');import conf_m102_a01_batch_00_solver as prior
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];x=s['train']['x'];y=s['train']['y'];T=s['train']['traj'];z,h=prior._coordinates(s,x)
def aff(x):return np.c_[np.ones(len(x)),x]
p=s['panel_b'];B=np.linalg.lstsq(aff(p['u']),p['v'],rcond=None)[0]
Ws=[x,z,h,aff(x)@B,aff(z)@B]; I=list(combinations_with_replacement(range(4),2)); Q=np.c_[aff(x),np.prod(x[:,I],axis=2)]
rng=np.random.default_rng(9); folds=[(np.arange(52),np.arange(52,72))]+[(a[:52],a[52:]) for a in [rng.permutation(72) for _ in range(3)]]
def features(p,kind,ix):
 X=x[ix]; A=Q[ix]; v=X@p[:4]+p[4]; U=aff(X)
 if kind=='divq':return A/(1+v[:,None]**2)
 if kind=='divsplit':return np.c_[U,A[:,5:]/(1+v[:,None]**2)]
 if kind=='divlin':return np.c_[A,U*v[:,None]**2/(1+v[:,None]**2)]
 if kind=='cub':return np.c_[A,v**3]
 if kind=='ratcub':return np.c_[A,v**3/(1+v*v)]
 if kind=='sin':return np.c_[A,np.sin(v)]
 if kind=='tanh':return np.c_[A,np.tanh(v)]
 if kind=='kink':return np.c_[A,abs(v)]
 if kind=='kinkq':return np.c_[A,abs(v)*v]
 if kind=='sinewarp':return np.c_[A,U*np.sin(v[:,None])]
 if kind=='ratq':return np.c_[A,(v**4/(1+v*v))]
 if kind=='cubicrank':return np.c_[A,v**3,(X@p[5:9]+p[9])**3]
 if kind=='triple':return np.c_[A,np.prod(U@p.reshape(3,5).T,axis=1)]
if __name__=='__main__':
 for kind in ['divq','divsplit','divlin','cub','ratcub','sin','tanh','kink','kinkq','ratq','sinewarp','cubicrank','triple']:
  out=[]
  for tr,te in folds:
   yt=y[tr]; rng=np.random.default_rng(813); opts=[]
   def obj(p):
    A=features(p,kind,tr); c=np.linalg.lstsq(A,yt,rcond=1e-9)[0]; return np.r_[A@c-yt,.003*p]
   fixed=[np.r_[w,.05] for w in np.vstack([np.eye(4),B[1:].T,np.linalg.inv(B[1:]).T])]; init=fixed+[rng.normal(0,.65,5) for i in range(10)]
   if kind in ['cubicrank','triple']:init=[np.tile(a,(2 if kind=='cubicrank' else 3,))+rng.normal(0,.4,10 if kind=='cubicrank' else 15) for a in init]
   for p in init:
    p=least_squares(obj,p,max_nfev=80,ftol=1e-7).x; opts.append((np.linalg.norm(obj(p)),p))
   opts.sort(key=lambda a:a[0]); best,p=opts[0];A=features(p,kind,tr);c=np.linalg.lstsq(A,yt,rcond=1e-9)[0];pred=features(p,kind,te)@c; out.append((np.mean(abs(pred-y[te])),np.sqrt(np.mean((A@c-yt)**2)),p.round(3).tolist()))
  print(kind,[(round(a,5),round(b,5)) for a,b,p in out],out[0][2],flush=True)
 c=np.linalg.lstsq(Q[:52],y[:52],rcond=None)[0];r=y-Q@c;ind=np.argsort(-abs(r));print('res extremes',np.c_[ind,x[ind],y[ind],r[ind]][:16].round(3));print('summary',s['train']['summary']);print('arch'); print(open(prior.__file__).read())
