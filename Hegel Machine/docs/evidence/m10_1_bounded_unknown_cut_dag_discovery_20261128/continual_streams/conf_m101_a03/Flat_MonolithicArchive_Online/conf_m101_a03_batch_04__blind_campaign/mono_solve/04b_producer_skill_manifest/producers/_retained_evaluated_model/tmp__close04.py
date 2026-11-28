from operator04 import s,Z,Y
from batch_solver import _affine as a,_fit as fit,_products as q
import numpy as np
from scipy.optimize import least_squares
u,v=s['panel_b']['u'],s['panel_b']['v']
rng=np.random.default_rng(795); folds=[np.arange(52,72),np.arange(56,72),*np.array_split(rng.permutation(72),4),np.argsort(abs(Z[:72]).max(1))[-16:]]
def metrics(pred,y):
 e=abs(pred-y); return [e[:,0].mean(),e[:,1:].mean(),e[:,-1].mean()]
def designs(x,b):
 d=np.c_[a(x),b]; return [d,np.c_[d,q(b)],np.c_[a(b),q(b)],np.c_[a(b),b[:,0]*b[:,3],b[:,1]*b[:,3]]]
def run(f,extra):
 def D(x,p):
  h=x*np.exp(p[:4])+(p[4:] if len(p)>4 else 0)
  z=np.c_[a(x),f(h)]; return z if extra is None else np.c_[z,extra(x)]
 def res(p):
  d=D(u,p); return (d@fit(d,v)-v).ravel()
 p=least_squares(res,np.r_[np.full(4,np.log(.87)),np.zeros(4)],bounds=(-3,3),max_nfev=100).x
 d=D(u,p); w=fit(d,v); e=d@w-v; lev=(d*np.linalg.pinv(d).T).sum(1); print('panel',np.sqrt(np.mean(e**2)),np.sqrt(np.mean((e/(1-lev[:,None]))**2)),p)
 return D(Z,p)@w
np.set_printoptions(precision=5,suppress=True)
for en,extra in [('none',None),('squares',lambda x:x*x),('adjacent',lambda x:x*np.roll(x,-1,axis=1))]:
 print(en)
 b=run(lambda x:x/(1+x*x),extra)
 for iy in (1,2,3):
  for it in (0,2):
   err=[]
   for j in folds:
    i=np.setdiff1d(np.arange(72),j); ds=designs(Z,b); dy,dt=ds[iy],ds[it]; t=Y[i,1:]; center=t.mean(0); _,_,vh=np.linalg.svd(t-center,full_matrices=False); axes=vh[:2].T
    py=dy[j]@fit(dy[i],Y[i,0]); pt=center+(dt[j]@fit(dt[i],(t-center)@axes))@axes.T
    err.append(metrics(np.c_[py,pt],Y[j]))
   print('readouts',iy,it,'52/20 and worst',np.r_[err[0],np.max(err,axis=0)])
