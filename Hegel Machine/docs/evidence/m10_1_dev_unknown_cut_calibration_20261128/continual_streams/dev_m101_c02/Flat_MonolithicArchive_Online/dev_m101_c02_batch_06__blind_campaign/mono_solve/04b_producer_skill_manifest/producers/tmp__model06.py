import numpy as np,pickle
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
ls=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]
def aff(z):return np.c_[np.ones(len(z)),z]
def poly(z,d): return np.column_stack([np.prod(z[:,i],axis=1) for k in range(d+1) for i in combinations_with_replacement(range(z.shape[1]),k)])
def calibration(s):
 u,v=s['panel_a']['u'],s['panel_a']['v']; keep=np.linalg.norm(v,axis=1)<5; u,v=u[keep],v[keep]; eye=np.eye(4)
 def design(w): return np.c_[np.tile(eye,(len(w),1)),(w[:,:,None]*eye).reshape(-1,4),w.ravel()**3,((w-1)*np.roll(w,-1,axis=1)).ravel()]
 p=ls(design(u),v.ravel()); print('A archive coef',np.round(p,6),'rms',np.std(design(u)@p-v.ravel()))
 def cal(x):
  w=(x-p[:4])/p[4:8]
  for _ in range(16):
   nxt=np.roll(w,-1,axis=1); err=(design(w)@p).reshape(-1,4)-x
   jac=(p[4:8]+3*p[8]*w*w+p[9]*nxt)[:,:,None]*eye+p[9]*(w-1)[:,:,None]*np.roll(eye,1,axis=1)
   w-=np.clip(np.linalg.solve(jac,err[...,None])[...,0],-.5,.5)
  return w
 return cal
if __name__=='__main__':
 tr=s['train']; x=tr['x']; y=tr['y']; t=tr['traj']; w=calibration(s)(x); print('latent ranges',w.min(0),w.max(0));
 def assess(z):
  a=z[:56]; b=z[56:]; pred=b@ls(a,np.c_[y,t][:56]); return np.mean(abs(pred[:,0]-y[56:])),np.mean(abs(pred[:,1:]-t[56:]))
 for key,z in [('raw',x),('cal',w)]:
  for degree in (1,2,3): print(key,degree,np.round(assess(poly(z,degree)),5))
 for key,k in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
  u,v=s[key]['u'],s[key][k]
  for label,f in [('diag2',lambda a:np.c_[aff(a),a*a]),('diag3',lambda a:np.c_[aff(a),a*a,a**3]),('poly2',lambda a:poly(a,2))]:
   a=f(u); coef=ls(a,v); h=(a*np.linalg.pinv(a).T).sum(1); z=f(w)@coef
   print(key,k,label,'panelLOO',round(np.sqrt(np.mean(((a@coef-v)/(1-h[:,None]))**2)),5),'downstream affine,interaction,quad',[np.round(assess(g),5) for g in (aff(z),np.c_[aff(z),z[:,0]*z[:,2]+z[:,1]*z[:,3]],poly(z,2))])
