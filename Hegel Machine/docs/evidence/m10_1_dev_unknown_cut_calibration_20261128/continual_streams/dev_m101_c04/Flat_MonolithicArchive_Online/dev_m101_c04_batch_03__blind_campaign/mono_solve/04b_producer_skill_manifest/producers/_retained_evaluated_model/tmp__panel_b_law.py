from grammar import *
from scipy.optimize import least_squares
p=s['panel_b']; u,v=p['u'],p['v']
A=abs
D=lambda x,t: np.sign(x)*np.maximum(A(x)-t,0)
models={
 'dead_power':(lambda x,p:D(x,p[0])*np.maximum(A(x)-p[0],1e-20)**(p[1]-1),[.25,1.3],[0,.6],[.8,4]),
 'poly_dead':(lambda x,p:D(x,p[0])*(1+p[1]*A(D(x,p[0]))),[.25,.4],[0,-.7],[.8,8]),
 'rational':(lambda x,p:x**3/(1+p[0]*x*x),[1],[0],[20]),
 'power_rational':(lambda x,p:x*A(x)**p[1]/(1+p[0]*A(x)**p[1]),[1,2],[0,.3],[20,7]),
 'exp_gate':(lambda x,p:x*(-np.expm1(-p[0]*A(x)**p[1])),[1,2],[.001,.3],[20,8]),
 'soft_dead_logistic':(lambda x,p:x/(1+np.exp(np.clip(-p[0]*(A(x)-p[1]),-500,500))),[5,.5],[.01,0],[100,2]),
 'shifted_cubic':(lambda x,p:x+p[0]*x**3,[1],[0],[1000]),
 'sine_difference':(lambda x,p:x-p[0]*np.sin(p[1]*x),[.35,2.8],[0,.3],[2,6]),
 'tanh_difference':(lambda x,p:x-p[0]*np.tanh(p[1]*x),[.7,1.4],[0,.1],[5,10]),
 'logcosh':(lambda x,p:x-np.tanh(p[0]*x)/p[0],[1],[.05],[12]),
 'threshold_cubic':(lambda x,p:np.sign(x)*np.maximum(A(x)**p[1]-p[0]**p[1],0),[.25,1.6],[0,.5],[.8,5]),
 'threshold_rational':(lambda x,p:D(x,p[0])*A(D(x,p[0]))**2/(1+p[1]*D(x,p[0])**2),[.1,1],[0,0],[.8,40]),
 'soft_dead_sqrt':(lambda x,p:(np.sqrt((x-p[0])**2+p[1]**2)-np.sqrt((x+p[0])**2+p[1]**2))/2+x,[.3,.1],[.01,0],[1.4,1]),
}
def get(name,i=None,orth=True):
 fun,init,low,high=models[name]; uu,vv=(u,v) if i is None else (np.delete(u,i,0),np.delete(v,i,0)); yc=vv-vv.mean(0)
 def profile(p, ret=False):
  f=fun(uu,p); off=f.mean(0); fc=f-off
  if orth:
   l,sv,r=np.linalg.svd(fc.T@yc); m=(l@r)*(sv.sum()/np.sum(fc**2))
  else:m=ls(fc,yc)
  if ret:return lambda x:(fun(x,p)-off)@m+vv.mean(0)
  return (fc@m-yc).ravel()
 opt=least_squares(profile,init,bounds=(low,high),gtol=1e-12,ftol=1e-11,max_nfev=150)
 return opt.x,np.std(opt.fun),profile(opt.x,True)
if __name__=='__main__':
 for name in models:
  for orth in (True,False):
   w,r,f=get(name,orth=orth); loo=np.sqrt(np.mean([(get(name,i,orth)[2](u[i:i+1])-v[i:i+1])**2 for i in range(len(u))])); print(name,orth,np.round(w,5),round(r,7),round(loo,7))
