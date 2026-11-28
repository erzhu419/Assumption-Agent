exec(open('/tmp/probe.py').read().split('U,D,V=')[0])
from scipy.special import erf
from scipy.optimize import minimize_scalar
from numpy.polynomial.legendre import legval

def measure(name,g,u=u,v=v,z=z):
 d=aff(g(u)); c=fit(d,v); w=g(z)@c[1:]+c[0]; r=d@c-v
 wy=np.c_[w,w[:,0]*w[:,2]+w[:,1]*w[:,3]]
 errs=[]
 for hold in [np.arange(52,72)]+[np.random.default_rng(i).permutation(72)[52:] for i in range(12)]:
  sel=np.setdiff1d(np.arange(72),hold); p=wy[hold]@fit(wy[sel],Y[sel,0]);
  f=-np.expm1(-np.outer(t,[.42,1.8])); a=fit(f,Y[sel,1:].T).T; p2=w[hold]@fit(w[sel],a)@f.T
  errs.append([abs(p-Y[hold,0]).mean(),abs(p2-Y[hold,1:]).mean()])
 print(name,'panel',np.std(r),'row residual med/max',np.median(np.linalg.norm(r,axis=1)),np.max(np.linalg.norm(r,axis=1)), '52/20',np.array(errs[0]),'worst',np.max(errs,axis=0))
 return r
print('B bounds',u.min(),u.max(),'A raw outliers',np.sort(np.linalg.norm(s['panel_a']['u']-s['panel_a']['v'],axis=1))[-6:]); print('telemetry',s['telemetry_vec'])
for deg in [3,5,7,9,11]:
 for odd in [True,False]:
  ker=np.polynomial.legendre.legvander
  order=list(range(3,deg+1,2)) if odd else list(range(2,deg+1)); sc=max(1.2,np.max(abs(u)))
  def warp(x,p):return x/sc+ker(x/sc,deg)[...,order]@p
  def res(p):
   d=aff(warp(u,p));return (d@fit(d,v)-v).ravel()
  p=least_squares(res,np.zeros(len(order)),max_nfev=250).x; measure((deg,odd,p),lambda x:warp(x,p))
for name,fn,start in [('sinerf',lambda x,p:x+p[0]*erf(p[1]*x)+p[2]*np.sin(p[3]*x),[-.647,1.47,.01,3.8]),('exp',lambda x,p:x+p[0]*x*np.exp(-p[1]*x*x),[-1.11,1.05]),('rational',lambda x,p:x+p[0]*x/(1+p[1]*x*x),[-1.08,.82]),('tanh3',lambda x,p:np.tanh(p[0]*x)+p[1]*x**3,[1.,1.]),('asinh',lambda x,p:x+p[0]*np.arcsinh(p[1]*x),[-1.,1.]),('atan',lambda x,p:x+p[0]*np.arctan(p[1]*x),[-.78,1.35]),('sin',lambda x,p:x+p[0]*np.sin(p[1]*x),[-.4,2.87]),('sin_cub',lambda x,p:x+p[0]*np.sin(p[1]*x)+p[2]*x**3,[-.4,2.87,0.]),('erf',lambda x,p:x+p[0]*erf(p[1]*x),[-.648,1.47])]:
 def res(p):
  d=aff(fn(u,p));return (d@fit(d,v)-v).ravel()
 p=least_squares(res,start,max_nfev=300).x
 r=measure((name,p),lambda x:fn(x,p)); print('  residual split',np.std(r[:12]),np.std(r[12:]))
