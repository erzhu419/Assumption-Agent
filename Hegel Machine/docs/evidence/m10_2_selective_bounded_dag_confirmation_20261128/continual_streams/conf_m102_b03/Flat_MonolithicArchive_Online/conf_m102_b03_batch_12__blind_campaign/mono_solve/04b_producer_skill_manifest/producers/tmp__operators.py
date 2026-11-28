import numpy as np,pickle
from scipy.optimize import least_squares, minimize_scalar
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; core=np.load('/tmp/core.npz'); w=core['w']; a=core['cal']; n=52
D=lambda x:np.c_[np.ones(len(x)),x]
ls=lambda x,y:np.linalg.lstsq(x,y,rcond=1e-10)[0]
poly=lambda q,deg: np.column_stack([np.prod(q[:,I],axis=1) for j in range(deg+1) for I in combinations_with_replacement(range(q.shape[1]),j)])
np.set_printoptions(precision=4,suppress=True)
print('core',a,core['rates'],w)
u,v=s['panel_a'].values();r=v-D(u)@a; good=np.linalg.norm(r,axis=1)<max(.003,5*np.median(np.linalg.norm(r,axis=1))); print('A residual',np.quantile(np.linalg.norm(r[good],axis=1),[0,.5,1]));
for deg in [1,2,3]:
 F=poly(u[good],deg); P=np.linalg.pinv(F); rr=(F@P@v[good]-v[good])/(1-np.sum(F*P.T,axis=1))[:,None]; print('A LOO',deg,np.std(rr),ls(F,v[good]) if deg==2 else '')
for j in range(4):
 U=u[good]; jj=(j+1)%4; f=D(np.c_[U[:,j],U[:,jj],U[:,j]*U[:,jj],U[:,j]**3]); M=ls(f,v[good,j]); print('A archived05 grammar',j,M,np.std(f@M-v[good,j]))
c=s['panel_c']; u=c['u']; P=ls(poly(u,2),c['v1']); print('C1 coefficients',P)
v=c['v2'];
for name,fn,k0 in [('cos',np.cos,2.15),('sin',np.sin,2.15),('swish',lambda x:x/(1+np.exp(-x)),4),('tanh',np.tanh,2.15),('softplus',lambda x:np.logaddexp(0,x),4)]:
 for withlin in [False,True]:
  for mode in [1,2,3]:
   def F(z,p):
    arg=z*(p[:4] if mode==3 else p[0])+(p[-4:] if mode>=2 else 0)
    return D(np.c_[z,fn(arg)] if withlin else fn(arg))
   def fun(p):
    f=F(u,p); return (f@ls(f,v)-v).ravel()
   p0=([k0]*4 if mode==3 else [k0])+([0.]*4 if mode>=2 else [])
   fit=least_squares(fun,p0,max_nfev=200); f=F(u,fit.x); pin=np.linalg.pinv(f); err=(f@pin@v-v)/(1-np.sum(f*pin.T,axis=1))[:,None];print('C2',name,withlin,mode,np.std(fit.fun),np.std(err),fit.x)
