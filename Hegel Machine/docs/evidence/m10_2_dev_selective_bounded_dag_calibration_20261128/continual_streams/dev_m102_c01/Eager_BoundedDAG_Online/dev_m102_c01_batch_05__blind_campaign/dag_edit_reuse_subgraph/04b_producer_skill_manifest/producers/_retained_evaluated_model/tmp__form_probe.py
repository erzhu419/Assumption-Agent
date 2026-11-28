from readout_probe import *
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
x=s['train']['x']; y=s['train']['y']; t=s['train']['traj']; tmp=copy.deepcopy(s); tmp['test']['x']=x; p=node_00.run_node({'__input__':tmp}); st=p['st'][:,:4]; sy=p['sy']; A=p['scores']; I=np.arange(52); J=np.arange(52,72)
print('correlations x, sy, st, state',np.round(np.corrcoef(np.c_[x,sy,st,A,y].T)[-1,:-1],3))
for name,Z in [('x',x),('xt',np.c_[x,st]),('xs',np.c_[x,sy]),('xta',np.c_[x,st,A]),('all',np.c_[x,st,sy,A])]:
 F=np.c_[np.ones(len(x)),Z]; ii,jj=np.triu_indices(Z.shape[1]); F2=np.c_[F,Z[:,ii]*Z[:,jj]]
 for kind,f in [('linear',F),('quadratic',F2)]:
  w=np.linalg.lstsq(f[I],y[I],rcond=1e-5)[0]; e=y-f@w; print(name,kind,np.std(e[I]),np.abs(e[J]).mean())
 R=F2 if name=='x' else F
 for fn in [np.sin,np.tanh,lambda z:z*z/(1+z*z),np.cos]:
  z=(Z-Z.mean(0))/np.maximum(Z.std(0),1e-8); d=z.shape[1]
  def fit(b,T=I,ret=False):
   g=np.c_[R,fn(z@b[:d]+b[-1])]; w=np.linalg.lstsq(g[T],y[T],rcond=1e-7)[0]; e=y-g@w
   return e if ret else e[T]
  rng=np.random.default_rng(0); outs=[]
  for rep in range(13):
   v=least_squares(fit,rng.normal(0,.5,d+1),max_nfev=160); e=fit(v.x,ret=True); outs.append((np.std(e[I]),np.abs(e[J]).mean(),v.x))
  e,h,b=min(outs,key=lambda el:el[0]); print(name,getattr(fn,'__name__',''),round(e,5),round(h,5),np.round(b,2))
