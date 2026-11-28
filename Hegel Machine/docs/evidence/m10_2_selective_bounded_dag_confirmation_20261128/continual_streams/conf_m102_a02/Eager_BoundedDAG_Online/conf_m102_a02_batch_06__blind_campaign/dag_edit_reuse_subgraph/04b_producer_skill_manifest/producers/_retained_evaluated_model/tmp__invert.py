import numpy as np,pickle
from itertools import combinations_with_replacement as cwr
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def cal(p,x):
 u,v=p['u'],p['v']; med=np.median(v,axis=0); dev=np.max(abs(v-med),axis=1); keep=dev < 2.8*np.median(dev); u,v=u[keep],v[keep]; c=[]
 for i in range(4):
  z,w=u[:,i],u[:,(i+1)%4]; c.append(np.linalg.lstsq(np.c_[np.ones(len(u)),z,w*(1+z),z**3],v[:,i],rcond=None)[0]);
 b,a,g,h=np.array(c).T; z=(x-b)/a; ix=np.arange(4); jx=(ix+1)%4
 for _ in range(25):
  w=z[:,jx]; r=b+a*z+g*w*(1+z)+h*z**3-x; J=np.zeros((len(z),4,4)); J[:,ix,ix]=a+g*w+3*h*z*z; J[:,ix,jx]=g*(1+z); step=np.linalg.solve(J,r[...,None])[...,0]; z-=step/np.maximum(1,np.max(abs(step),axis=1,keepdims=True));
 print('inverse residual',abs(r).max(),'cal rows',len(u),'cal coefficients',np.array(c)); return z
z=cal(s['panel_a'],np.r_[s['train']['x'],s['test']['x']]); Y=np.c_[s['train']['y'],s['train']['traj']]; z=z[:len(Y)]; N=52

def poly(x,d):return np.column_stack([np.prod(x[:,j],axis=1) for k in range(d+1) for j in cwr(range(4),k)])
def eval_(name,A):
 A=A[:len(z)]; B=np.linalg.lstsq(A[:N],Y[:N],rcond=None)[0]; E=abs(A@B-Y); print(name,A.shape[1],np.round(np.r_[E[:N,0].mean(),E[:N,1:].mean(),E[N:,0].mean(),E[N:,1:].mean(),E[N:,-1].mean()],5)); return B
for d in range(1,5):
 A=poly(z,d); B=eval_('latent poly'+str(d),A)
 if d<=3: print('terms',[(k,np.round(v[[0,1,-1]],5)) for k,v in zip([j for i in range(d+1) for j in cwr(range(4),i)],B)])
for f in (np.sin,np.tanh,np.arctan):
 for d in (1,2,3):eval_(f.__name__+str(d),poly(f(z),d))
for strength in (.15,.35,.6,1.,1.5,2.):
 for d in (1,2,3):eval_('rational '+str((strength,d)),poly(z/(1+strength*z*z),d))
print('latent range',z.min(0),z.max(0)); print('latent correlations',np.corrcoef(np.c_[z,Y].T)[:4,4:]);
