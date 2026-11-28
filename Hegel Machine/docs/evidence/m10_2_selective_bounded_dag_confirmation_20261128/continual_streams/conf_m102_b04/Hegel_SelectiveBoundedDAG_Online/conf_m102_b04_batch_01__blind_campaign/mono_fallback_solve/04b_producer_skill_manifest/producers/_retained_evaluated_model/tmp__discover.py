from operators import *
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
np.set_printoptions(precision=4,suppress=True)
p=s['panel_c']; u=p['u']; v=p['v1']; w=p['v2']; b=fit(basis(s['panel_b']['u'],1),s['panel_b']['v']); c=np.load('/tmp/cal.npy');
def cal(x):return (x-c[0])/c[1:].diagonal()
def rot(x):return basis(x,1)@b

def profile(D,Y):
 e=loo(D,Y);return np.sqrt(np.mean(e*e,axis=0))
for name,q in [('raw',u),('B',rot(u)),('cal',cal(u)),('Bcal',rot(cal(u)))]:
 for name2,D in [('square',np.c_[basis(q,1),q*q]),('diag cubes',np.c_[basis(q,1),q*q,q**3]),('cross',basis(q,2)),('pairs',np.c_[basis(q,1), q*np.roll(q,-1,axis=1)]),('sine',np.c_[np.ones(len(q)),np.sin(q),q*q]),('tanh',np.c_[np.ones(len(q)),np.tanh(q),q*q])]:print(name,name2,profile(D,v),profile(D,w))
for fn in [np.sin,np.tanh,lambda x:x/np.sqrt(1+x*x),np.arctan,lambda x:x]:
 for target in [v,w]:
  def calc(l):
   D=np.c_[np.ones(len(u)),fn(l*u)/l,u*u];return profile(D,target)
  print(str(fn),[calc(l).mean() for l in (.25,.5,.8,1,1.25,1.5,2)])

def gp(q,Y,n,kern,rank=False):
 xx=q[:n]; yc=Y[:n]; ym=yc.mean(0); sd=yc.std(0); yc=(yc-ym)/sd
 D=(q[:,None,:]-xx[None,:,:])**2
 def K(th):
  d=np.sqrt(np.maximum(D@np.exp(th[:q.shape[1]]),1e-15));
  if kern=='rbf':return np.exp(-.5*d*d)
  if kern=='m52':return (1+d+d*d/3)*np.exp(-d)
  if kern=='rq':return (1+.5*d*d)**-1
 def obj(th,ret=False):
  k=K(th); a=k[:n]+np.eye(n)*np.exp(th[-1]); ch=cho_factor(a); al=cho_solve(ch,yc)
  if ret:return (k@al)*sd+ym
  return .5*np.sum(yc*al)+yc.shape[1]*np.log(np.diag(ch[0])).sum()
 r=minimize(obj,np.r_[np.zeros(q.shape[1]),-7.],bounds=[(-8,7)]*q.shape[1]+[(-13,-2)],method='L-BFGS-B');return obj(r.x,True),r.x
x=s['train']['x']; z=s['train']['traj']; V=np.linalg.svd(z[:52],full_matrices=False)[2][:1]; amp=z@V.T; y=s['train']['y'][:,None];
for name,q in [('x',x),('cal',cal(x)),('b',rot(x)),('bcal',rot(cal(x)))]:
 for k in ('rbf','m52','rq'):
  for t in (y,amp):
   pred,pars=gp(q,t,52,k);print('gp',name,k,'y' if t is y else 'amp',np.mean(abs(pred[52:]-t[52:])),pars)
