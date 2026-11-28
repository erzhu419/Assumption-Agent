import numpy as np,pickle,itertools
from scipy.optimize import least_squares, minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];np.set_printoptions(precision=5,suppress=True,linewidth=140)
def col(*x):return np.column_stack(x)
def fit(X,Y):return np.linalg.lstsq(X,Y,rcond=1e-11)[0]
def bases(u):
 return {'linear':u,'square':u*u,'sin':np.sin(u),'cos':np.cos(u),'tanh':np.tanh(u),'cube':u**3,'cross':col(*(u[:,i]*u[:,j] for i,j in itertools.combinations(range(4),2)))}
links={'tanh':np.tanh,'sin':np.sin,'atan':np.arctan,'erf':__import__('scipy').special.erf,'asinh':np.arcsinh,'sat':lambda z:z/np.sqrt(1+z*z)}
for p,j in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 u=s[p]['u'];v=s[p][j]; X=bases(u);one=np.ones(len(u)); results=[]
 for name,f in links.items():
  for extras in [(),('linear',),('square',),('cos',),('cross',),('tanh',),('sin',)]:
   if name in extras:continue
   for mode in [1,4,8]:
    def feature(q,u,X):
     scale=np.exp(q[:min(4,mode)]); bias=q[4:] if mode==8 else 0
     return col(np.ones(len(u)),f(u*scale+bias),*(X[k] for k in extras))
    def obj(q):
     M=feature(q,u,X); return (M@fit(M,v)-v).ravel()
    lo=np.r_[np.full(min(4,mode),-2.3),np.full(max(0,mode-4),-1.5)];hi=np.r_[np.full(min(4,mode),1.5),np.full(max(0,mode-4),1.5)]
    o=least_squares(obj,np.zeros(mode),bounds=(lo,hi),max_nfev=150);M=feature(o.x,u,X);w=fit(M,v);P=np.linalg.pinv(M); r=M@w-v; loo=r/(1-(M*P.T).sum(1))[:,None];results.append((np.sqrt(np.mean(r*r)),np.sqrt(np.mean(loo*loo)),name,extras,mode,np.exp(o.x[:min(4,mode)]),o.x[4:],w))
 print(p,j)
 for k in [0,1]:
  for z in sorted(results,key=lambda e:e[k])[:5]:print(z[:-1]);print(z[-1] if k==1 else '')
T=s['train']['traj'];t=s['eval_times'];one=np.ones(len(t)); f=lambda r:-np.expm1(-r*t)
for name,m,fn in [('exp_t',1,lambda p:col(f(p[0]),t)),('exp_log',1,lambda p:col(f(p[0]),np.log1p(t))),('exp_exp',2,lambda p:col(f(p[0]),f(p[1]))),('exp_exp1',1,lambda p:col(f(p[0]),f(.22))),('exp_one',1,lambda p:col(f(p[0]),one)),('exp_sin',2,lambda p:col(f(p[0]),np.sin(p[1]*t))),('exp_dexp',1,lambda p:col(f(p[0]),t*np.exp(-p[0]*t)))]:
 def residual(p):
  B=fn(p);return (B@fit(B,T.T)-T.T).ravel()
 o=least_squares(residual,[1.25,.25][:m],bounds=(.0001,5));print(name,o.x,np.sqrt(np.mean(o.fun**2)),np.mean(abs(o.fun)))
