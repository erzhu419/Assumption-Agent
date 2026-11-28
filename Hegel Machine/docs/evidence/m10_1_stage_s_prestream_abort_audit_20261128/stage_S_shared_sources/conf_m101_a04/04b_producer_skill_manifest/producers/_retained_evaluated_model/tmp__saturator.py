import contextlib,io
with contextlib.redirect_stdout(io.StringIO()): exec(open('/tmp/graph.py').read())
fns={
 'abs_power':(lambda x,p:x/(1+p[0]*abs(x)**p[1])**p[2],[1.,1.,1.],[.01,.35,.1],[12,4,5]),
 'abs_power_lin':(lambda x,p:x/(1+p[0]*abs(x)**p[1])**p[2]+p[3]*x,[1.,1.,1.,0],[.01,.35,.1,-5],[12,4,5,5]),
 'exp_abs':(lambda x,p:np.sign(x)*(-np.expm1(-p[0]*abs(x)**p[1]))+p[2]*x,[2.,1.,0],[.05,.35,-5],[12,4,5]),
 'logpow':(lambda x,p:np.sign(x)*np.log1p(p[0]*abs(x)**p[1])+p[2]*x,[2.,1.,0],[.05,.35,-5],[12,4,5]),
 'tanh_rational':(lambda x,p:np.tanh(p[0]*x)/(1+p[1]*x*x),[2.,.2],[.05,-.6],[12,10]),
 'atan_rational':(lambda x,p:np.arctan(p[0]*x)/(1+p[1]*x*x),[2.,.2],[.05,-.6],[12,10]),
 'pade_abs':(lambda x,p:x*(1+p[0]*abs(x))/(1+np.exp(p[1])*abs(x)+np.exp(p[2])*x*x),[.2,0.,0.],[-2,-8,-8],[8,5,5]),
 'pade_square':(lambda x,p:x*(1+p[0]*x*x)/(1+np.exp(p[1])*x*x+np.exp(p[2])*x**4),[.2,0.,0.],[-2,-8,-8],[8,5,5]),
}
from sklearn.model_selection import KFold
for name,(fun,init,lo,hi) in fns.items():
 def g(x,p):return aff(fun(x,p))
 def opt(i):
  def r(p):
   e=g(U[i],p);return (e@fit(e,R[i])-R[i]).ravel()
  o=least_squares(r,init,bounds=(lo,hi),max_nfev=400);return o.x,fit(g(U[i],o.x),R[i]),np.std(o.fun)
 pp=np.empty_like(R)
 for i,j in KFold(4,shuffle=True,random_state=61).split(U):
  p,w,_=opt(i);pp[j]=g(U[j],p)@w
 p,w,e=opt(np.arange(len(U))); print(name,np.mean(abs(pp-R)),e,p,np.linalg.svd(w[1:])[1])
