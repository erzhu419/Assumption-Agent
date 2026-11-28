import numpy as np
from scipy.optimize import minimize_scalar

def ls(a,b): return np.linalg.lstsq(a,b,rcond=None)[0]
def amap(s,verbose=False):
 x,y=(s['panel_a'][k] for k in ('u','v')); X=np.c_[np.ones(len(x)),x]; keep=np.linalg.norm(y-X@ls(X,y),axis=1)<4; x,y,X=x[keep],y[keep],X[keep]; u=np.roll(x,-1,axis=1); v=x
 funcs=[lambda u,v,k:u*v,lambda u,v,k:u*v/(1+k*(u*u+v*v)),lambda u,v,k:u*v/(1+k*(u*v)**2),lambda u,v,k:np.sin(k*u*v),lambda u,v,k:np.tanh(k*u*v),lambda u,v,k:np.arctan(k*u*v),lambda u,v,k:(np.cos(k*(u-v))-1),lambda u,v,k:(np.cos(k*(u+v))-1),lambda u,v,k:np.sin(k*u)*np.sin(k*v),lambda u,v,k:u*v/(1+k*(u-v)**2),lambda u,v,k:u*v/(1+k*(u+v)**2)]
 def fit(fn,k,full=False):
  H=fn(u,v,k); C=np.array([ls(np.c_[X,H[:,j]],y[:,j]) for j in range(4)]).T; err=np.mean((X@C[:5]+H*C[5]-y)**2); return (err,C) if full else err
 fitlist=[]
 for i,fn in enumerate(funcs):
  r=minimize_scalar(lambda k:fit(fn,k),bounds=(.001,3),method='bounded'); err,c=fit(fn,r.x,True); fitlist.append((err,i,r.x,c));
  if verbose: print('A',i,r.x,err**.5)
 _,i,k,c=min(fitlist,key=lambda a:a[0]); fn=funcs[i];
 if verbose: print('A coefficients',c)
 def inv(y):
  z=(y-c[0])@np.linalg.inv(c[1:5]);
  for it in range(25):
   u=np.roll(z,-1,axis=1); h=fn(u,z,k); eps=1e-5; da=(fn(u+eps,z,k)-fn(u-eps,z,k))/(2*eps); db=(fn(u,z+eps,k)-fn(u,z-eps,k))/(2*eps); J=np.broadcast_to(c[1:5].T,(len(z),4,4)).copy(); j=np.arange(4); J[:,j,j]+=db*c[5]; J[:,j,(j+1)%4]+=da*c[5]; r=c[0]+z@c[1:5]+h*c[5]-y; d=np.linalg.solve(J,r[...,None])[...,0]; z-=d/np.maximum(1,np.linalg.norm(d,axis=1,keepdims=True))
  if verbose: print('A inverse max residual',np.max(abs(c[0]+z@c[1:5]+fn(np.roll(z,-1,axis=1),z,k)*c[5]-y)))
  return z
 return inv

def other(s,z,verbose=False):
 p=s['panel_b']; b=np.c_[np.ones(len(p['u'])),p['u']]; bc=ls(b,p['v']);
 p=s['panel_c']; u=p['u']; v=p['v1']; x=np.c_[np.ones(len(u)),u];
 funcs={'rat':lambda x,k:x/(1+k*x*x),'tanh':lambda x,k:np.tanh(k*x),'pair':lambda x,k:(x+np.roll(x,1,axis=1))/(1+k*(x+np.roll(x,1,axis=1))**2),'cycpair':lambda x,k:x*np.roll(x,1,axis=1)/(1+k*(x*x+np.roll(x,1,axis=1)**2))}
 def fit(fn,k):
  a=np.c_[x,fn(u,k)]; c=ls(a,v); return np.mean((a@c-v)**2),c
 opts=[]
 for name,fn in funcs.items():
  r=minimize_scalar(lambda k:fit(fn,k)[0],bounds=(.01,4),method='bounded'); opts.append((r.fun,name,r.x,fit(fn,r.x)[1]));
  if verbose: print('C1',name,r.x,r.fun**.5)
 _,name,k,c=min(opts,key=lambda a:a[0]);
 def err2(k):
  a=np.c_[np.ones(len(u)),np.cos(k*u)]; c=ls(a,p['v2']); return np.mean((a@c-p['v2'])**2),c
 r=minimize_scalar(lambda k:err2(k)[0],bounds=(1.5,3.2),method='bounded'); c2=err2(r.x)[1];
 return {'z':z,'b':bc[0]+z@bc[1:],'v1':np.c_[np.ones(len(z)),z,funcs[name](z,k)]@c,'v2':np.c_[np.ones(len(z)),np.cos(r.x*z)]@c2}
if __name__=='__main__':
 import pickle
 s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; z=amap(s,True)(np.r_[s['train']['x'],s['test']['x']]); print('ranges',z.min(0),z.max(0)); other(s,z,True)
