import numpy as np,pickle,itertools
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; y=s['train']['y']; T=s['train']['traj']; z=np.load('/tmp/stages.npz')['b']; n=56
P=np.array(list(itertools.combinations_with_replacement(range(4),2))); Q=np.prod(z[:,P],axis=2); F=np.c_[np.ones(len(z)),z,Q]; U=np.c_[np.ones(len(z)),z]
def ls(a,b):return np.linalg.lstsq(a,b,rcond=None)[0]
def mat(c):
 m=np.zeros((4,4)); i,j=P.T; m[i,j]=m[j,i]=c/(1+(i!=j));return m
D=np.linalg.svd(F[:n],full_matrices=False); resid=y[:n]-F[:n]@ls(F[:n],y[:n]); G=np.prod(z[:,list(itertools.combinations_with_replacement(range(4),3))],axis=2); w=ls(np.c_[F[:n],G[:n]],y[:n]); print('cubic coefficients',np.round(w[15:],4)); print('quadratic y eig', np.linalg.eigh(mat(w[5:15]))[0]); print('quadratic state eigvectors',np.linalg.eigh(mat(ls(F[:n],T[:n])[5:,-1]))); print('state vs y quadratic correlation', np.corrcoef(ls(F[:n],T[:n])[5:,-1],w[5:15])[0,1]); print('noise y residual order',np.sort(abs(resid))[-8:]);
funcs={ 'sin':lambda a:np.sin(a),'tanh':np.tanh,'atan':np.arctan,'softsat':lambda a:a/np.sqrt(1+a*a),'rat':lambda a:a/(1+a*a),'exp':lambda a:np.exp(np.clip(a,-15,15)), 'cos':np.cos,'sinh':lambda a:np.sinh(np.clip(a,-15,15)) }
rng=np.random.default_rng(13); v=rng.normal(size=(12,4)); v=np.r_[np.eye(4),ls(F[:n],y[:n])[None,1:5],v]; v/=np.linalg.norm(v,axis=1,keepdims=True)
for name in [*funcs,'divide','linear_divide','quad_divide']:
 mode=0 if name in funcs else {'divide':1,'linear_divide':2,'quad_divide':3}[name]
 def basis(p):
  if mode==3:
   q=p[0]+Q@p[1:];return F/(1+q[:,None]**2)
  a=U@p
  if mode==1:return F/(1+a[:,None]**2)
  if mode==2:return np.c_[F,U/(1+a[:,None]**2)]
  return np.c_[F,funcs[name](a)]
 def res(p):
  a=basis(p)[:n];return a@ls(a,y[:n])-y[:n]
 starts=[np.r_[bias,g*k] for g in v for bias,k in [(0.,.8),(.3,1.8)]] if mode!=3 else [np.r_[.2,rng.normal(0,.25,10)] for _ in range(12)]
 fits=[least_squares(res,p,max_nfev=85) for p in starts]; fit=min(fits,key=lambda f:f.cost); a=basis(fit.x); w=ls(a[:n],y[:n]); err=a@w-y
 print(name, 'MAEs',*(round(np.mean(abs(e)),6) for e in [err[:n],err[n:]]),'p',np.round(fit.x,4),'amp',np.round(w[-1],4))
