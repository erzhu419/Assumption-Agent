import pickle,numpy as np
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; u=s['panel_c']['u']; z=s['panel_c']['v2']; np.set_printoptions(precision=4,suppress=True)
pairs=[(i,j) for i in range(4) for j in range(i+1,4)]
a=np.c_[np.ones(len(u)),u]; uv=np.array([u[:,i]*u[:,j] for i,j in pairs]).T
ls=lambda f,v:np.linalg.lstsq(f,v,rcond=1e-10)[0]
def assess(f,v):
 w=ls(f,v); e=v-f@w; h=np.sum(np.linalg.qr(f,mode='reduced')[0]**2,axis=1);return np.sqrt(np.mean(e**2)), np.sqrt(np.mean((e/(1-h)[:,None])**2))
for key,v in [('v1',s['panel_c']['v1']),('v2',z)]:
 print(key,'affine',assess(a,v),'weights',ls(a,v),'resid sv',np.linalg.svd(v-a@ls(a,v),compute_uv=False)); print('quadratic',assess(np.c_[a,u*u,uv],v)); print('per row affine residual',np.linalg.norm(v-a@ls(a,v),axis=1))
 d={'x/(1+k|x|)':lambda k:u/(1+k*abs(u)),'x²/(1+kx²)':lambda k:u*u/(1+k*u*u),'x/(1+kx²)':lambda k:u/(1+k*u*u),'tanh(kx)':lambda k:np.tanh(k*u),'sin(kx)':lambda k:np.sin(k*u),'atan(kx)':lambda k:np.arctan(k*u),'x²/(1+k|x|)':lambda k:u*u/(1+k*abs(u)),'exp(-kx²)':lambda k:np.exp(-k*u*u),'x3/(1+kx²)':lambda k:u**3/(1+k*u*u),'linear radial':lambda k:u/(1+k*np.sum(u*u,axis=1,keepdims=True)),'quadratic radial':lambda k:u*u/(1+k*np.sum(u*u,axis=1,keepdims=True)), 'x²/(1+kx⁴)':lambda k:u*u/(1+k*u**4)}
 for name,fn in d.items():
  fit=lambda k:assess(np.c_[a,fn(k)],v)[0]; o=minimize_scalar(fit,bounds=(0.005,15),method='bounded'); f=np.c_[a,fn(o.x)];print(name,round(o.x,4),assess(f,v))
