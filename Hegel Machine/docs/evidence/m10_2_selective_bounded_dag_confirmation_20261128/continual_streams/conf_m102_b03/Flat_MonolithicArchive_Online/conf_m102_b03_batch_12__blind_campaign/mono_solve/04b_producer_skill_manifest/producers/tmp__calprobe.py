import numpy as np,pickle
from scipy.optimize import minimize_scalar,least_squares
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; a=np.load('/tmp/core.npz')['cal']; u=s['panel_a']['u']; v=s['panel_a']['v']; D=lambda x:np.c_[np.ones(len(x)),x]; good=np.linalg.norm(D(u)@a-v,axis=1)<1; u=u[good];v=v[good]; ls=lambda x,y:np.linalg.lstsq(x,y,rcond=1e-10)[0];print('n',len(u))
def err(f,y):
 p=np.linalg.pinv(f); b=p@y; e=f@b-y; return np.sqrt(np.mean((e/np.maximum(1-np.sum(f*p.T,axis=1),1e-6))**2)),np.std(e),b
for j in range(4):
 z=u[:,j]; o=u[:,(j+2)%4];y=v[:,j]; entries=[]
 for i in range(4):
  for k in range(i,4): entries.append((f'{i}{k}',u[:,i]*u[:,k]));
 for i,k,l in combinations_with_replacement(range(4),3):entries.append((f'{i}{k}{l}',u[:,i]*u[:,k]*u[:,l]));
 for i in range(4):
  for name,f in [('sin',np.sin),('tanh',np.tanh),('cos',np.cos),('atan',np.arctan),('softsign',lambda x:x/(1+abs(x))),('rational',lambda x:x/(1+x*x))]:
   for scale in [.6,1.,1.25,1.8,2.15,2.4,3.]: entries.append((f'{name}{i}@{scale}',f(scale*u[:,i])))
 print('channel',j)
 F=D(u); pool=entries[:];
 for steps in range(7):
  best=min(range(len(pool)),key=lambda i:err(np.c_[F,pool[i][1]],y)[0]); name,x=pool.pop(best); F=np.c_[F,x]; E=err(F,y); print(name,np.round(E[:2],7),np.round(E[2],4));
 for mode in range(3):
  q=np.c_[z,o];
  def ff(k):return D(np.c_[u,np.sin(q*k)] if mode==0 else np.c_[u,np.tanh(q*k)] if mode==1 else np.c_[u,np.cos(q*k)] )
  def rr(k):
   f=ff(k);return f@ls(f,y)-y
  p=least_squares(rr,[1.25,1.25],bounds=(.05,6.)).x; print('profile',mode,p,err(ff(p),y))
