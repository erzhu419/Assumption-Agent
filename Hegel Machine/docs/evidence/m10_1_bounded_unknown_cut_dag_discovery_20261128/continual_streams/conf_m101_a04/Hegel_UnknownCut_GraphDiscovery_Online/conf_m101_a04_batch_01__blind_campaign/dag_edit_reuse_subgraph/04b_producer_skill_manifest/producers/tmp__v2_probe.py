import numpy as np,pickle
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['panel_c']['u']; y=s['panel_c']['v2']
print('panel bounds',x.min(0),x.max(0)); print('train bounds',s['train']['x'].min(0),s['train']['x'].max(0)); print('test bounds',s['test']['x'].min(0),s['test']['x'].max(0))
funcs={'cos':lambda x,a:(1-np.cos(x*a))/(a*a), 'cosh':lambda x,a:(np.cosh(x*a)-1)/(a*a), 'rational':lambda x,a:x*x/(1+a*x*x), 'log':lambda x,a:np.log1p(a*x*x)/a, 'exp':lambda x,a:-np.expm1(-a*x*x)/a, 'sqrt':lambda x,a:x*x/(1+np.sqrt(1+a*x*x)), 'exp_asym':lambda x,a:(np.expm1(x*a)-x*a)/(a*a), 'sin':lambda x,a:(x*a-np.sin(x*a))/(a**3)}
def estimate(f,x,y,hi,shared):
 def basis(x,a): return np.c_[np.ones(len(x)),x,f(x,a)]
 def residual(a):
  b=basis(x,a); return (b@np.linalg.lstsq(b,y,rcond=None)[0]-y).ravel()
 a=least_squares(residual,np.full(1 if shared else 4,min(.6,hi/2)),bounds=(.001,hi),ftol=1e-11,gtol=1e-11).x
 return a,basis,np.linalg.lstsq(basis(x,a),y,rcond=None)[0]
for deg in [2,3,4,5]:
 m=np.c_[np.ones(len(x)),*[x**i for i in range(1,deg+1)]]; p=np.linalg.pinv(m); e=y-m@(p@y); h=(m*p.T).sum(1); print('additive polynomial',deg,'rms',np.sqrt(np.mean(e**2)),'loo',np.sqrt(np.mean((e/(1-h[:,None]))**2)))
 if deg==4: print('quadratic to quartic coefficients', (p@y)[5:].round(4))
for name,f in funcs.items():
 for shared in [True,False]:
  hi=4 if name in ('cos','cosh','sin') else 3
  a,b,w=estimate(f,x,y,hi,shared); e=y-b(x,a)@w; loo=[]
  for i in range(len(x)):
   m=np.arange(len(x))!=i; aa,bb,ww=estimate(f,x[m],y[m],hi,shared); loo.append(y[i]-bb(x[i:i+1],aa)[0]@ww)
  print(name,shared,'param',a.round(4),'rms',np.sqrt(np.mean(e**2)).round(7),'worst loo',np.sqrt(np.mean(np.array(loo)**2,0)).max().round(7))
