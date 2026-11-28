import pickle,numpy as np
from itertools import combinations_with_replacement
from scipy.optimize import minimize_scalar
from scipy.special import expit
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];tr=s['train'];ls=lambda a,b:np.linalg.lstsq(a,b,rcond=1e-10)[0];D=lambda x:np.c_[np.ones(len(x)),x];rm=lambda r:np.mean(r*r)**.5
u,v=[s['panel_a'][k] for k in ('u','v')];rng=np.random.default_rng(11)
for j in range(4):
 a=u[:,j];b=u[:,(j+2)%4];
 for order in (2,3,4,5):
  m=[q for o in range(1,order+1) for q in combinations_with_replacement(range(2),o)];x=D(np.column_stack([np.prod(np.c_[a,b][:,q],axis=1) for q in m]));t=v[:,j];best=min((ls(x[i],t[i]) for i in (rng.choice(len(u),len(m)+3,False) for _ in range(400))),key=lambda c:np.median(abs(x@c-t)));r=abs(x@best-t);g=r<max(.001,4*np.median(r));c=ls(x[g],t[g]);print('cal',j,order,g.sum(),rm(x[g]@c-t[g]),[(q,round(w,5)) for q,w in zip([()]+m,c) if abs(w)>.001]);
  if rm(x[g]@c-t[g])<.0005:break
u=s['panel_c']['u'];w=np.roll(u,-1,axis=1);pairs=np.column_stack([u[:,i]*u[:,j] for i in range(4) for j in range(i+1,4)]);f={'softsign':lambda a,k:a/(1+k*abs(a)), 'rational':lambda a,k:a/(1+k*a*a),'tanh':lambda a,k:np.tanh(a*k),'sin':lambda a,k:np.sin(k*a),'cos':lambda a,k:np.cos(k*a),'softplus':lambda a,k:np.logaddexp(0,k*a),'swish':lambda a,k:a*expit(k*a),'gauss':lambda a,k:np.exp(-k*a*a),'relu':lambda a,k:np.maximum(0,a-k)}
for key in ('v1','v2'):
 y=s['panel_c'][key];candidates=[]
 for base,xx in [('none',u[:,:0]),('lin',u),('lin3',np.c_[u,u**3]),('adj',u*w),('linadj',np.c_[u,u*w]),('linOpp',np.c_[u,u*np.roll(u,2,axis=1)[:,:]]) ,('linPair',np.c_[u,pairs])]:
  candidates.append((rm(D(xx)@ls(D(xx),y)-y),base,'none','',0))
  for name,fn in f.items():
   for zname,z in [('u',u),('adj',u*w),('opp',u*np.roll(u,2,axis=1))]:
    def err(k):
     x=D(np.c_[xx,fn(z,k)]);return rm(x@ls(x,y)-y)
    p=minimize_scalar(err,bounds=(.03,3.25),method='bounded');candidates.append((p.fun,base,name,zname,p.x))
 print(key,sorted(candidates)[:18])
