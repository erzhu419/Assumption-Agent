from panels import *
from scipy.optimize import differential_evolution
from scipy.special import erf
p=s['panel_b']; u,v=p['u'],p['v']
def orth(z,v):
 a,b=z.mean(0),v.mean(0); z=z-a; w=v-b; P,D,Q=np.linalg.svd(z.T@w); M=(P@Q)*(D.sum()/np.sum(z*z)); return M,b-a@M
F={
 'shrink':(lambda x,t:np.sign(x)*np.maximum(abs(x)-t,0),[(0,.8)]),
 'shrinkp':(lambda x,t,p:np.sign(x)*np.maximum(abs(x)-t,0)**p,[(0,.8),(.5,3)]),
 'shrinklin':(lambda x,t,k:np.sign(x)*np.maximum(abs(x)-t,0)+k*x,[(0,.8),(-.8,.7)]),
 'softshrink':(lambda x,t,k,b:(np.logaddexp(0,k*(x-t))-np.logaddexp(0,k*(-x-t)))/k+b*x,[(0,.8),(1,40),(-.8,.3)]),
 'tanh':(lambda x,a,b:x-b*np.tanh(a*x),[(.3,5),(.1,1.4)]),
 'erf':(lambda x,a,b:x-b*erf(a*x),[(.3,5),(.1,1.4)]),
 'rat':(lambda x,a,b:x-b*x/(1+a*x*x),[(.01,8),(.1,2)]),
 'sqrt':(lambda x,a,b:x-b*x/np.sqrt(1+a*x*x),[(.01,8),(.1,2)]),
 'gauss':(lambda x,a,b:x-b*x*np.exp(-a*x*x),[(.01,8),(.1,2)]),
 'expcusp':(lambda x,a,b:x+b*np.sign(x)*np.expm1(-a*abs(x)),[(.01,10),(.1,3)]),
 'sin':(lambda x,a,b:x-b*np.sin(a*x),[(.3,5),(.1,2)]),
 'atan':(lambda x,a,b:x-b*np.arctan(a*x),[(.3,8),(.1,2)]),
 'abs':(lambda x,a,b:x-b*x/(1+a*abs(x)),[(.01,8),(.1,2)]),
 'ratpow':(lambda x,a,b,c:x-b*x/(1+a*x*x)**c,[(.01,8),(.1,2),(.12,5)]),
 'piece2':(lambda x,t,b:x+b*np.sign(x)*np.minimum(abs(x),t)**2,[(.05,1.3),(-5,0)]),
 'poly':(lambda x,a,b,c,d:a*x+x**3+b*x**5+c*x**7+d*x**9,[(-1,1),(-2,2),(-2,2),(-2,2)]),
 'polyabs':(lambda x,a,b,c,d:a*x+x*abs(x)+b*x**3+c*x**3*abs(x)+d*x**5,[(-2,2),(-4,4),(-4,4),(-4,4)])}
def run(key):
 f,bd=F[key]
 def res(x):
  z=f(u,*x); m,b=orth(z,v);return (z@m+b-v).ravel()
 o=differential_evolution(lambda x:np.mean(res(x)**2),bd,seed=4,tol=1e-7,popsize=10);m,b=orth(f(u,*o.x),v);return np.sqrt(o.fun),o.x,m,b
if __name__=='__main__':
 out={key:run(key) for key in F}
 for k,(e,a,m,b) in sorted(out.items(),key=lambda kv:kv[1][0]): print(k,e,np.round(a,6),np.linalg.norm(m[0]))
 k,(e,a,m,b)=min(out.items(),key=lambda kv:kv[1][0]); z=(v-b)@np.linalg.inv(m); j=np.argsort(abs(u.ravel())); print(np.round(np.c_[u.ravel(),z.ravel(),(z-F[k][0](u,*a)).ravel()][j][::2],5)); np.savez('/tmp/best_b.npz',k=k,a=a,m=m,b=b)
