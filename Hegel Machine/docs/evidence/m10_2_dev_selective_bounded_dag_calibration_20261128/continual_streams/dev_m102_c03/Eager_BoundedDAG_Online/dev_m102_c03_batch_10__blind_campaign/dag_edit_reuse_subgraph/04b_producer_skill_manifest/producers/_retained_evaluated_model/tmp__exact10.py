import numpy as np
from scipy.optimize import minimize_scalar

def ls(A,Y): return np.linalg.lstsq(A,Y,rcond=None)[0]
def latent(s,x):
 p=s['panel_a']; u,v=p['u'],p['v']; H=np.c_[np.ones(len(u)),u]; keep=np.linalg.norm(v-H@ls(H,v),axis=1)<4; u,v,H=u[keep],v[keep],H[keep]
 def f(k,ret=False):
  w=np.sin(k*u)-k*u; t=u*np.roll(u,-1,axis=1); C=np.stack([ls(np.c_[H,t[:,j],w[:,j]],v[:,j]) for j in range(4)],1); e=H@C[:5]+t*C[5]+w*C[6]-v
  return (C,np.std(e)) if ret else np.mean(e*e)
 k=minimize_scalar(f,bounds=(.05,.6),method='bounded').x; c,e=f(k,True)
 z=(x-c[0])@np.linalg.inv(c[1:5]); j=np.arange(4)
 for _ in range(20):
  w=np.roll(z,-1,axis=1); r=c[0]+z@c[1:5]+c[5]*z*w+c[6]*(np.sin(k*z)-k*z)-x; J=np.broadcast_to(c[1:5].T,(len(z),4,4)).copy(); J[:,j,j]+=c[5]*w+c[6]*k*(np.cos(k*z)-1);J[:,j,(j+1)%4]+=c[5]*z; d=np.linalg.solve(J,r[...,None])[...,0];z-=d/np.maximum(1,np.linalg.norm(d,axis=1,keepdims=True))
 return z,(k,e,c)

def maps(s,z):
 p=s['panel_b']; u,v=p['u'],p['v']; h=np.c_[np.ones(len(u)),u]; b=ls(h,v)
 p=s['panel_c'];u=p['u'];h=np.c_[np.ones(len(u)),u]; ones=np.ones(len(z));
 def f(k,fun,key):
  F=np.c_[h,fun(u,k)] if key=='v1' else np.c_[h[:,0],fun(u,k)]; C=ls(F,p[key]);return np.mean((F@C-p[key])**2),C
 fun=lambda x,k:x/(1+k*x*x); k=minimize_scalar(lambda k:f(k,fun,'v1')[0],bounds=(.1,3),method='bounded').x; _,c=f(k,fun,'v1'); _,d=f(2.4,lambda x,k:np.cos(k*x),'v2');return (np.c_[ones,z]@b,np.c_[ones,z,fun(z,k)]@c,np.c_[ones,np.cos(2.4*z)]@d),(b,c,d,np.std(h@b-v) if len(v)==len(h) else 0)
