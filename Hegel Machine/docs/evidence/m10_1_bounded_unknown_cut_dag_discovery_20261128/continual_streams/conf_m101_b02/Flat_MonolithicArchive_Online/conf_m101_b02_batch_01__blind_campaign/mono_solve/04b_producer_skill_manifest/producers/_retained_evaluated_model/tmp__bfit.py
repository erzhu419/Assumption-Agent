from panels import *
from scipy.special import erf
from scipy.optimize import differential_evolution
p=s['panel_b'];u=p['u'];v=p['v'];print('range', u.min(0),u.max(0));
def design(x):return np.c_[np.ones(len(x)),x]
F={
 'damped':(lambda x,a:x*np.exp(-a*x*x),[(.001,4)]),
 'dampedshift':(lambda x,a,b:x*np.exp(-a*(x-b)**2),[(.001,4),(-.5,.5)]),
 'rational':(lambda x,a,b:x/(1+a*x*x)**b,[(.001,6),(.07,8)]),
 'sinlin':(lambda x,a,b:np.sin(a*x)+b*x,[(.1,5),(-5,5)]),
 'sinrat':(lambda x,a,b:np.sin(a*x)/(1+b*x*x),[(.1,5),(-.4,4)]),
 'tanhlin':(lambda x,a,b:np.tanh(a*x)+b*x,[(.1,5),(-5,5)]),
 'erflin':(lambda x,a,b:erf(a*x)+b*x,[(.1,5),(-5,5)]),
 'tanhshift':(lambda x,a,b:np.tanh(a*(x-b)),[(.1,5),(-.5,.5)]),
 'sinshift':(lambda x,a,b:np.sin(a*(x-b)),[(.1,5),(-.5,.5)]),
 'oddpoly':(lambda x,a,b,c:x+a*x**3+b*x**5+c*x**7,[(-4,1),(-4,4),(-4,4)]),
 'poly':(lambda x,a,b,c,d:x+d*x**2+a*x**3+b*x**5+c*x**7,[(-4,1),(-4,4),(-4,4),(-.2,.2)]),
 'sinpluscubic':(lambda x,a,b:np.sin(a*x)+b*x**3,[(.2,5),(-5,5)]),
 'tanhpluscubic':(lambda x,a,b:np.tanh(a*x)+b*x**3,[(.2,5),(-5,5)]),
 'clip':(lambda x,a:np.clip(x,-a,a),[(.05,.95)]),
 'oddexp':(lambda x,a:-np.sign(x)*np.expm1(-a*abs(x)),[(.05,4)]),
 'gausslin':(lambda x,a,b:x*(np.exp(-a*x*x)+b),[(.01,4),(-3,3)]),
 'softsignshift':(lambda x,a,b:(x-b)/(1+a*abs(x-b)),[(.01,6),(-.5,.5)])}
def fit_b(u,v,key):
 f,bounds=F[key];
 def obj(p):
  g=design(f(u,*p)); return np.mean((g@ls(g,v)-v)**2)
 opt=differential_evolution(obj,bounds,seed=8,tol=1e-8,popsize=10);b=ls(design(f(u,*opt.x)),v)
 return opt.x,b,np.sqrt(opt.fun)
if __name__=='__main__':
 for k in F:
  par,b,e=fit_b(u,v,k);print(k,np.round(par,5),e,np.linalg.svd(b[1:],compute_uv=False))
  if k=='oddpoly':
   z=(v-b[0])@np.linalg.inv(b[1:]); ii=np.argsort(u.ravel());print(np.round(np.c_[u.ravel(),z.ravel(),(z-F[k][0](u,*par)).ravel()][ii][::3],5))
