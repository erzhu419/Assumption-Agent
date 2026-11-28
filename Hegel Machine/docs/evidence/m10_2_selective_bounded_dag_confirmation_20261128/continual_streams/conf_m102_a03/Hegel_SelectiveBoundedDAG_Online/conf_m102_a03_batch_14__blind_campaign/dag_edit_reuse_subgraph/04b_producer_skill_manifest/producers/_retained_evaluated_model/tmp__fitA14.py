exec(open('/tmp/probe14.py').read().split("u=s['panel_a']")[0])
from scipy.optimize import least_squares
u,v=[s['panel_a'][k] for k in ('u','v')]; _,r=fit(u,v,terms[:15],True); good=r<5*np.median(r)
u,v=u[good],v[good]; j=np.arange(4); p=u[:,(j+2)%4]
for mode in ('rational','sin','tanh','arctan','cubic'):
 for extra in (0,1,2):
  def design_a(z,j,h):
   x=z[:,j]; w=z[:,(j+2)%4]; a=z[:,(j+1)%4]; b=z[:,(j-1)%4]; l=h[0];
   with np.errstate(over='ignore'):
    g={'rational':lambda:w/(1+l*w*w),'sin':lambda:np.sin(l*w),'tanh':lambda:np.tanh(l*w),'arctan':lambda:np.arctan(l*w),'cubic':lambda:w**3}[mode]()
   cols=[np.ones(len(z)),x,x**3,g];
   if mode=='cubic': cols+=[w]
   if extra==1: cols += [a,b,x*w,a*b]
   if extra==2: cols += [a,b,x*w,a*b,x*x,w*w,x*a,x*b,w*a,w*b]
   return np.column_stack(cols)
  def res(h):
   return np.column_stack([a@np.linalg.lstsq(a,v[:,j],rcond=None)[0]-v[:,j] for j in range(4) for a in [design_a(u,j,h)]])
  o=min([least_squares(lambda h:res(h).ravel(),[q],bounds=(.03,7)) for q in (.2,1.,2.,3.)],key=lambda r:r.cost); print(mode,extra,o.x, np.sqrt(np.mean(o.fun**2))); 
  if np.sqrt(np.mean(o.fun**2))<.022: print(np.array([np.linalg.lstsq(design_a(u,j,o.x),v[:,j],rcond=None)[0] for j in range(4)]).round(3))
