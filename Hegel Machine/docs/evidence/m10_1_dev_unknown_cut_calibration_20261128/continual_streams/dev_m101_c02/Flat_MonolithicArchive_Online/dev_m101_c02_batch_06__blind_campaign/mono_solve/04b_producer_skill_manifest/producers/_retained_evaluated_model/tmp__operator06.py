from model06 import *
from scipy.optimize import minimize_scalar
from sklearn.model_selection import KFold
u,v=(s['panel_b'][k] for k in ('u','v')); tr=s['train']; w=calibration(s)(tr['x']); y=tr['y']; T=tr['traj']; t=s['eval_times']; folds=[(np.arange(56),np.arange(56,72)),*KFold(4,shuffle=True,random_state=4).split(w)]
sv=np.linalg.svd(T,full_matrices=False); print('times',t,'time mode',sv[2][0]);
for name,tf in [('log',lambda p:np.log1p(p*t)),('exp',lambda p:-np.expm1(-p*t)),('rat',lambda p:t/(1+p*t)),('asinh',lambda p:np.arcsinh(p*t))]:
 def r(p):
  f=tf(p); return np.sqrt(np.mean((T-(T@f/(f@f))[:,None]*f)**2))
 m=minimize_scalar(r,bounds=(.001,15),method='bounded'); print('time',name,m.x,m.fun)
def score(z):
 a=aff(z); cross=z[:,0]*z[:,2]+z[:,1]*z[:,3]; sy=np.c_[a,cross]; both=[]
 for i,j in folds:
  pred=sy[j]@ls(sy[i],y[i]); traj=a[j]@ls(a[i],T[i]); both.append([np.mean(abs(pred-y[j])),np.mean(abs(traj-T[j]))]);
 return np.round(both,5)
def report(name,f,init=None,bounds=None):
 def resid(p,x,y):
  a=f(x,p); return (a@ls(a,y)-y).ravel()
 def opt(x,y):
  if init is None:return None
  options=[least_squares(resid,p,args=(x,y),bounds=bounds,max_nfev=180) for p in init]; return min(options,key=lambda q:q.cost).x
 p=opt(u,v); a=f(u,p); c=ls(a,v); e=np.sqrt(np.mean((a@c-v)**2));
 loo=[]
 for k in range(len(u)):
  i=np.arange(len(u))!=k; pp=opt(u[i],v[i]); loo.extend((f(u[k:k+1],pp)@ls(f(u[i],pp),v[i])-v[k]).ravel())
 print(name,'params',p,'panel rms/loo',round(e,6),round(np.std(loo),6),'56/16 and worst',score(f(w,p)@c)[0],score(f(w,p)@c).max(0),flush=True)
 if name=='diag cubic':print('diag cubic coefficient ratios',np.round(c[1:5]/c[9:13],3),'square coefficients',np.round(c[5:9],3));
 if name.startswith('rational'): print('operator',np.round(c,4));
report('diag cubic',lambda x,p:np.c_[aff(x),x**2,x**3]);
report('odd cubic unrestricted',lambda x,p:np.c_[aff(x),x**3]);
report('shared cubic',lambda x,p:aff(x**3+p[0]*x),[[0],[-.5]],([-8],[8]));
report('rational fixed2',lambda x,p:aff(x**3/(1+p[0]*x*x)+p[1]*x),[[.2,0],[1,-.1],[3,-.3]],([0,-6],[15,6]));
report('rational free power',lambda x,p:aff(x**3/(1+p[0]*abs(x)**p[1])+p[2]*x),[[.3,3,-.1],[1,2,-.2],[1,4,-.3]],([0,.5,-6],[15,9,6]));
report('sine plus linear',lambda x,p:aff(np.sin(p[0]*x)+p[1]*x),[[1,-1],[3,-1],[6,0],[9,0]],([.1,-20],[15,20]));
report('shared quintic',lambda x,p:aff(x**3+p[0]*x+p[1]*x**5),[[0,0],[-.3,-.3]],([-8,-8],[8,8]));
