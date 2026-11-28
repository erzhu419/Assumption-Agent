import sys
sys.path.insert(0,'/holdout_task')
from node_00 import _inverse,_polys
from panels import *
from scipy.optimize import least_squares
U,V=[s['panel_b'][k] for k in ('u','v')]; u=_inverse(s['panel_a'],s['train']['x']); z=s['train']['traj']; t=s['eval_times']; q=-np.expm1(-.62*t); Y=np.c_[s['train']['y'],z@q/(q@q)];
gs={'rat3':(lambda x,a:x**3/(1+a[0]*x*x),[.8]),'rat4':(lambda x,a:x**3/(1+a[0]*x*x+a[1]*x**4),[.8,.1]),'rat6':(lambda x,a:x**3/(1+a[0]*x*x+a[1]*x**4+a[2]*x**6),[.8,.1,.01]),'pow':(lambda x,a:x**3/(1+a[0]*x*x)**a[1],[.8,1]),'tanhcube':(lambda x,a:np.tanh(a[0]*x)**3,[.8]),'sincube':(lambda x,a:np.sin(a[0]*x)**3,[.8]),'log':(lambda x,a:np.sign(x)*np.log1p(a[0]*abs(x)**a[1]),[1.3,3.2]),'ratmix':(lambda x,a:x**3/(1+a[0]*x*x)+a[1]*np.sin(a[2]*x),[.8,.02,3]),'abspow':(lambda x,a:x**3/(1+a[0]*abs(x)**a[1]),[1,2]),'pureodd':(lambda x,a: np.column_stack([x**3,x**5,x**7]),[1])}
res=[]
for n,(g,a0) in gs.items():
 for e in ['none','lin','square','cross','triple','odd3','even','lin_sin','log']:
  def f(x,a):
   extra={'none':[], 'lin':[x], 'square':[x,x*x], 'cross':[x,features(x)['cross']], 'triple':[x,features(x)['triple']], 'odd3':[x,x**3], 'even':[x,x*x,x**4], 'lin_sin':[x,np.sin(2*x)],'log':[x,np.log1p(.365*x*x)]}[e]
   return np.column_stack([np.ones(len(x)),g(x,a)]+extra)
  def err(a):
   F=f(U,a); p=np.linalg.pinv(F);return ((F@(p@V)-V)/np.maximum(.02,1-(F*p.T).sum(1))[:,None]).ravel()
  lo=np.full(len(a0),.0001); hi=np.full(len(a0),8.); 
  if n=='ratmix':lo[1]=-2
  o=least_squares(err,a0,bounds=(lo,hi),max_nfev=70); F=f(U,o.x); c=np.linalg.lstsq(F,V,rcond=None)[0]; b=f(u,o.x)@c; res.append((np.mean(err(o.x)**2)**.5,n,e,o.x,b,c))
res.sort(key=lambda x:x[0]);print('panel fits');
for v in res[:14]:print(*v[:4]);
print('best coef',res[0][-1].round(4));
np.savez('/tmp/last9.npz',bs=np.array([v[4] for v in res[:14]]),u=u,Y=Y)
for v in res[:14]:
 b=v[4]; P=_polys(b,3); H=np.linalg.pinv(P[:52]); c=H@Y[:52]; E=Y-P@c; print('poly',v[1:3],np.mean(abs(E[:52]),0).round(4),np.mean(abs(E[52:]),0).round(4));
