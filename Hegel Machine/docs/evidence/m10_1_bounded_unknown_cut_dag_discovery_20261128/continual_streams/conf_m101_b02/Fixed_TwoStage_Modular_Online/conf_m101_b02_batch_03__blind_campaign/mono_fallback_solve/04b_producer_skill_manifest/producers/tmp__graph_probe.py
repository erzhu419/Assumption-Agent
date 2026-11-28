import pickle,numpy as np
from itertools import combinations
from scipy.optimize import least_squares, minimize_scalar
from sklearn.preprocessing import PolynomialFeatures
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; N=52
x=s['train']['x'];y=s['train']['y'];T=s['train']['traj'];t=s['eval_times'];Y=np.c_[y,T]
def D(u):return np.c_[PolynomialFeatures(2).fit_transform(u),u**3]
a=s['panel_a'];F=D(a['u']);v=a['v'];W=np.linalg.lstsq(F,v,rcond=None)[0]
for _ in range(50):
 q=np.maximum(np.linalg.norm(v-F@W,axis=1),.00025)**-.5;W=np.linalg.lstsq(q[:,None]*F,q[:,None]*v,rcond=None)[0]
m=np.linalg.norm(v-F@W,axis=1)<.005;W=np.linalg.lstsq(F[m],v[m],rcond=None)[0];z=D(x)@W
print('panel_a inliers, coef',m.sum(),np.round(W,4)); print('calibrated range',z.min(0),z.max(0))
b=s['panel_b'];B=np.linalg.lstsq(np.c_[np.ones(len(b['u'])),b['u']],b['v'],rcond=None)[0]
for name,u in [('x',x),('a',z),('b(x)',x@B[1:]+B[0]),('b(a)',z@B[1:]+B[0])]:
 for order in [1,2,3]:
  f=PolynomialFeatures(order).fit_transform(u); co=np.linalg.lstsq(f[:N],Y[:N],rcond=None)[0]; e=np.abs(Y-f@co)
  print(name,order,'train',e[:N,0].mean(),e[:N,1:].mean(),'valid',e[N:,0].mean(),e[N:,1:].mean())
 def fit(f):
  c=np.linalg.lstsq(f[:N],Y[:N],rcond=None)[0];r=Y[N:]-f[N:]@c;return np.mean(abs(r[:,0])),np.mean(abs(r[:,1:]))
 for name2,f in [('oddexp',lambda u,k:np.sign(u)*(-np.expm1(-k*abs(u)))),('tanh',lambda u,k:np.tanh(k*u)),('rational',lambda u,k:u/(1+k*u*u)),('sine',lambda u,k:np.sin(k*u))]:
  for k in [.5,1.,1.5]:
   h=f(u,k);g=np.c_[np.ones(len(u)),h,h*h,np.array([h[:,i]*h[:,j] for i,j in combinations(range(4),2)]).T];print(name,name2,k,np.round(fit(g),4))
for name,fn,p0 in [('two relax',lambda p: -np.expm1(-np.outer(t,p)),[.5,2]),('relax+time',lambda p: np.c_[-np.expm1(-p[0]*t),t],[1]),('relax+transient',lambda p: np.c_[-np.expm1(-p[0]*t),t*np.exp(-p[1]*t)],[1,1]),('damped osc',lambda p: np.c_[1-np.exp(-p[0]*t)*np.cos(p[1]*t),np.exp(-p[0]*t)*np.sin(p[1]*t)],[1,1])]:
 def resid(p):
  U=fn(p);return (U@np.linalg.lstsq(U,T[:N].T,rcond=None)[0]-T[:N].T).ravel()
 o=least_squares(resid,p0,bounds=(.001,20));U=fn(o.x); H=np.linalg.lstsq(U,T.T,rcond=None)[0].T;print(name,o.x,np.sqrt(np.mean(o.fun**2)), 'val projection',np.mean(abs(T[N:]-H[N:]@U.T)), 'amplitude corr',np.corrcoef(np.c_[z,y,H].T)[-2:,:-2])
