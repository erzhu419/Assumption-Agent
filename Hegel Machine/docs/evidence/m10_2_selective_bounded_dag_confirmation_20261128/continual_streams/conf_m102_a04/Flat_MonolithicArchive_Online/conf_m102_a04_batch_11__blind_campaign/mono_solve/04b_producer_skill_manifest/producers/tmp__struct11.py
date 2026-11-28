import numpy as np, pickle, itertools as it
from scipy.optimize import least_squares,minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def one(f):return np.c_[np.ones(len(f)),f]
def ls(f,y):return np.linalg.lstsq(f,y,rcond=1e-10)[0]
def fit(fn,u,y,q,bounds=(-np.inf,np.inf)):
 def r(q):
  f=fn(u,q);return (f@ls(f,y)-y).ravel()
 p=least_squares(r,q,bounds=bounds,gtol=1e-13,ftol=1e-12,xtol=1e-12).x; f=fn(u,p); w=ls(f,y); e=y-f@w; h=np.einsum('ij,ji->i',f,np.linalg.pinv(f)); return p,w,np.mean(e**2)**.5,np.mean((e/((1-h) if y.ndim==1 else (1-h)[:,None]))**2)**.5
if __name__=='__main__':
 a=s['panel_a']; keep=np.linalg.norm(a['v'],axis=1)<5; u,v=a['u'][keep],a['v'][keep];
 for i in range(4):
  z=u[:,[i,(i+2)%4]]
  for kind,g,q,b in [('sin',lambda x,q:np.sin(q*x),2.5,(.2,5)),('tanh',lambda x,q:np.tanh(q*x),1.2,(.1,4)),('rat',lambda x,q:x/(1+q*x*x),.6,(.01,4)),('soft',lambda x,q:x/(1+q*abs(x)),.8,(.01,4))]:
   for mode in [0,1,2]:
    def fn(z,q):
     x,y=z.T;return one(np.c_[x,x**3,g(y,q[0]),*([] if mode==0 else [y,x*y]),*([] if mode<2 else [x*x,y*y,x**5])])
    p,w,e,l=fit(fn,z,v[:,i],[q],b);print(i,kind,mode,p.round(5),w.round(5),round(e,7),round(l,7))
 p=s['panel_c'];u=p['u'];
 for key,fun,q,b in [('v1',lambda w,q:one(w/(1+q[0]*abs(w))+q[1]*w),[.8,.2],([.05,-2],[5,2])),('v2rat',lambda w,q:np.c_[one(w),w*w,w*w/(1+q[0]*w*w)],[.25],(.01,2)),('v2cos',lambda w,q:np.c_[one(w),w*w,np.cos(q[0]*w)],[2.5],(.1,5))]:
  k=key[:2]; pp,ww,e,l=fit(fun,u,p[k],q,b); print(key,pp,e,l,ww.round(6))
 t=s['eval_times']; Y=s['train']['traj'].T; f=lambda t,q:-np.expm1(-t[:,None]*q); q,w,e,l=fit(f,t,Y,[.4,1.8],(.01,8));print('TIME',q,e,l,'amp std',w.std(1),'y correlations',np.corrcoef(np.vstack([w,s['train']['y']]))[-1])
