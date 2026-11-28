import numpy as np,pickle,itertools
np.set_printoptions(precision=4,suppress=True,linewidth=160)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];u,v=[s['panel_a'][k] for k in ('u','v')]; print(np.c_[u,v]); rng=np.random.default_rng(17)
for i in range(4):
 p=u[:,i]; q=u[:,(i+1)%4]; y=v[:,i]; print('coord',i)
 zz={'aff':np.c_[p,q], 'old':np.c_[p,q,p*q,p**3], 'full pair':np.c_[p,q,p*q,p**3,q**3,p*p,q*q,p*p*q,p*q*q], 'all diag':np.c_[u,u**3,p*q], 'sin':np.c_[p,q,p*q,np.sin(2.15*p)],'tanh':np.c_[p,q,p*q,np.tanh(2.15*p)],'rational':np.c_[p,q,p*q,p/(1+2.15*p*p)],'atan':np.c_[p,q,p*q,np.arctan(2.15*p)],'denom q':np.c_[p,q,p*q,p/(1+2.15*q*q)]}
 for name,z in zz.items():
  z=np.c_[np.ones(len(z)),z]; n=z.shape[1]; ix=np.array([rng.choice(len(z),n,replace=False) for _ in range(650)]); b=np.linalg.pinv(z[ix])@y[ix,:,None] if False else (np.linalg.pinv(z[ix])@y[ix][...,None])[...,0]; e=np.abs(z@b.T-y[:,None]); j=np.argmin(np.median(e,axis=0)); ee=e[:,j]; keep=ee<max(0.001,3*np.median(ee)); c=np.linalg.lstsq(z[keep],y[keep],rcond=None)[0]; e0=abs(z@c-y); print(name, np.quantile(e0,[.5,.8,.9,1]),np.count_nonzero(e0<.001),c)
