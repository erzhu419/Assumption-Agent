from probe import *
from scipy.optimize import least_squares
u=s['panel_a']['u'];v=s['panel_a']['v'];r=np.roll(u,-1,axis=1)
for name,fun,p0 in [('cubic',lambda x,p:x+p[0]*x**3,[-.1]),('rat',lambda x,p:x/(1+p[0]*x*x),[.1]),('tanh',lambda x,p:np.tanh(p[0]*x),[.5]),('sin',lambda x,p:np.sin(p[0]*x),[.5]),('poly',lambda x,p:x+p[0]*x**3+p[1]*x**5,[-.1,.01]),('power',lambda x,p:x/(1+p[0]*abs(x)**p[1]),[.22,2]),('root',lambda x,p:x/(1+p[0]*abs(x)**p[1])**p[2],[.22,2,.5])]:
 for mix in range(3):
  h=np.zeros((len(u),4,11));j=np.arange(4);h[:,j,j]=1;h[:,:,-3:]=np.stack([r,u*r,u],-1)
  def des(p):
   h[:,j,4+j]=fun(u,p);h[:,:,-3]=r if mix==0 else fun(r,p);h[:,:,-2]=u*r if mix<2 else fun(u,p)*fun(r,p);return h
  keep=np.ones(len(u),bool)
  for i in range(5):
   def resid(p):
    H=des(p)[keep].reshape(-1,11);return (H@fit(H,v[keep].ravel())-v[keep].ravel())
   opt=least_squares(resid,p0,max_nfev=70); H=des(opt.x);coef=fit(H[keep].reshape(-1,11),v[keep].ravel());e=np.linalg.norm(H@coef-v,axis=1); keep=e<5*np.median(e)
  print('A',name,mix,opt.x,keep.sum(),np.std(e[keep]),np.median(e))
u=s['panel_c']['u'];v=s['panel_c']['v2']
for name,fun,p0 in [('poly',lambda x,p:x+p[0]*x*x+p[1]*x**3+p[2]*x**4,[.45,0,0]),('rat',lambda x,p:(x+p[0]*x*x)/(1+p[1]*x*x),[.45,0]),('exp',lambda x,p:np.expm1(p[0]*x),[.8]),('link',lambda x,p:x+p[0]*(1-np.cos(p[1]*x)),[.5,1]),('root',lambda x,p:x+p[0]*(np.sqrt(1+p[1]*x*x)-1),[.5,1]),('power',lambda x,p:x+p[0]*abs(x)**p[1],[.45,2]),('softplus',lambda x,p:np.logaddexp(0,p[0]*x+p[1]),[1,0]),('flex',lambda x,p:(x+p[0]*x*x)/(1+p[1]*abs(x)**p[2]),[.45,.05,2])]:
 def rr(p):
  H=aff(fun(u,p));return (H@fit(H,v)-v).ravel()
 o=least_squares(rr,p0,max_nfev=150);print('C2',name,o.x,np.std(o.fun))
