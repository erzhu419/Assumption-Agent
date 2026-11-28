from probe import *
from scipy.optimize import least_squares
u=s['panel_c']['u'];v=s['panel_c']['v2']
for name,fun,p0 in [('poly',lambda x,p:x+p[0]*x*x+p[1]*x**3+p[2]*x**4,[.45,0,0]),('rat',lambda x,p:(x+p[0]*x*x)/(1+p[1]*x*x),[.45,0]),('exp',lambda x,p:np.expm1(p[0]*x),[.8]),('link',lambda x,p:x+p[0]*(1-np.cos(p[1]*x)),[.5,1]),('root',lambda x,p:x+p[0]*(np.sqrt(1+p[1]*x*x)-1),[.5,1]),('power',lambda x,p:x+p[0]*abs(x)**p[1],[.45,2]),('softplus',lambda x,p:np.logaddexp(0,p[0]*x+p[1]),[1,0]),('flex',lambda x,p:(x+p[0]*x*x)/(1+p[1]*abs(x)**p[2]),[.45,.05,2])]:
 def rr(p):
  H=aff(fun(u,p));return (H@fit(H,v)-v).ravel()
 o=least_squares(rr,p0,max_nfev=150);print('C2',name,o.x,np.std(o.fun))
