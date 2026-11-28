import pickle,numpy as np
from scipy.optimize import least_squares, minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
x=s['panel_c']['u']; n,d=x.shape
funcs={'sin':np.sin,'cos':np.cos,'tanh':np.tanh,'gauss':lambda z:np.exp(-z*z),'evenrat':lambda z:1/(1+z*z),'oddrat':lambda z:z/(1+z*z),'abs':np.abs}
def solve(x,y,fun,k,affine,shift=0):
    f=np.column_stack((np.ones(len(x)),x if affine else np.empty((len(x),0)),fun(x*k+shift)))
    b=np.linalg.lstsq(f,y,rcond=None)[0]; r=f@b-y
    return f,b,r
for key in ['v1','v2']:
 y=s['panel_c'][key]; results=[]
 for name,fun in funcs.items():
  for affine in [False,True]:
   obj=lambda k:np.mean(solve(x,y,fun,k,affine)[2]**2)
   grids=np.linspace(.25,6,116); i=np.argmin([obj(g) for g in grids]); t=minimize_scalar(obj,bounds=(max(.1,grids[i]-.07),grids[i]+.07),method='bounded'); k=t.x
   f,b,r=solve(x,y,fun,k,affine); q=np.linalg.qr(f)[0]; loo=r/(1-(q*q).sum(1))[:,None]
   results.append((np.sqrt(t.fun),name,affine,k,np.max(np.sqrt((loo**2).mean(0)))))
 print(key,'shared scale:')
 for a in sorted(results): print(' ', a)
 print(key,'axis scales/offsets:')
 for _,name,affine,k,_ in sorted(results)[:5]:
  fun=funcs[name]
  obj=lambda p:solve(x,y,fun,p[:d],affine,p[d:] if len(p)>d else 0)[2].ravel()
  for shift in [False,True]:
   p0=np.r_[np.repeat(k,d),np.zeros(d)] if shift else np.repeat(k,d)
   opt=least_squares(obj,p0,max_nfev=150,bounds=(np.r_[np.repeat(.15,d),np.repeat(-2,len(p0)-d)], np.r_[np.repeat(8,d),np.repeat(2,len(p0)-d)]))
   print(' ',name,affine,shift,round(np.sqrt(np.mean(opt.fun**2)),7),np.round(opt.x,4))
a=s['panel_a']; F=np.c_[np.ones(len(a['u'])),a['u']]; v=a['v']; b=np.linalg.lstsq(F,v,rcond=None)[0]
for _ in range(12):
 r=np.linalg.norm(F@b-v,axis=1); w=np.sqrt(np.minimum(1,1.5*np.median(r)/np.maximum(r,1e-15))); b=np.linalg.lstsq(F*w[:,None],v*w[:,None],rcond=None)[0]
r=np.linalg.norm(F@b-v,axis=1); keep=r<5*np.median(r); b=np.linalg.lstsq(F[keep],v[keep],rcond=None)[0]; print('calibration inliers',keep.sum(),'max residual',np.linalg.norm((F@b-v)[keep],axis=1).max(),'B',np.round(b,5))
