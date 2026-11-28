import numpy as np,pickle,itertools
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; y=tr['y']; T=tr['traj']; d=np.load('/tmp/stages.npz'); n=56; t=s['eval_times']; print('times',t)
def ls(f,y):return np.linalg.lstsq(f,y,rcond=None)[0]
def feats(x,deg):
 i=[p for k in range(deg+1) for p in itertools.combinations_with_replacement(range(4),k)];return np.stack([np.prod(x[:,p],axis=1) for p in i],axis=1),i
def stat(a):return np.round([np.mean(abs(a)),np.sqrt(np.mean(a*a)),np.max(abs(a))],6)
for k in ['z','b']:
 x=d[k]; f,idx=feats(x,2); w=ls(f[:n],T[:n]); print(k,'state coeff',np.round(w[:,[0,3,7]],5)); print('y coeff',np.round(ls(f[:n],y[:n]),5))
 for j in [0,3,7]:
  m=np.zeros((4,4));
  for (a,b),v in zip(idx[5:],w[5:,j]):m[a,b]=m[b,a]=v/(1+(a!=b))
  print('state quadratic eigen',j,np.linalg.eigh(m)[0])
 for degree in [1,2,3,4]:
  f,_=feats(x,degree); a=f[:n]; b=f[n:]; p=ls(a,y[:n]); print(k,degree,'y train/test',stat(a@p-y[:n]),stat(b@p-y[n:]),'state',stat(b@ls(a,T[:n])-T[n:]))
 f,_=feats(x,2); g=f[:,1:]; yt=y[:n]; a=f[:n];
 for gg in [x,x*x,g[:,4:],g]:
  G=gg[:n]; q=ls(np.c_[a,-yt[:,None]*G],yt); print('linearized rational',gg.shape[1], 'denom',np.round(q[15:],4), 'min',min(1+gg@q[15:]),'cv',stat((f@q[:15]/(1+gg@q[15:])-y)[n:]))
 for form in ['rational','tanh','softsign','sin','atan','exponential','asinh','softsat']:
  def phi(z):
   return {'rational':lambda:z/(1+z*z),'tanh':lambda:np.tanh(z),'softsign':lambda:z/(1+abs(z)),'sin':lambda:np.sin(z),'atan':lambda:np.arctan(z),'exponential':lambda:np.exp(np.clip(z,-20,20)),'asinh':lambda:np.arcsinh(z),'softsat':lambda:z/np.sqrt(1+z*z)}[form]()
  def basis(z):return np.c_[np.ones(len(z)),z,phi(z)]
  def res(p):
   b=basis(a@p);return b@ls(b,yt)-yt
  p0=ls(a,yt); fits=[least_squares(res,j*p0,max_nfev=110) for j in [.25,.65,1.2,2.0]]; r=min(fits,key=lambda q:q.cost); b=basis(f@r.x); print('score',form,stat(r.fun),stat((b@ls(b[:n],yt)-y)[n:]))
