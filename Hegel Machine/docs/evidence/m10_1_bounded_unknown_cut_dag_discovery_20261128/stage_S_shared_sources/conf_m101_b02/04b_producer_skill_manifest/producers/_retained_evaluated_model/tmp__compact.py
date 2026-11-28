import sys,pickle,numpy as np
from scipy.optimize import least_squares,minimize_scalar
sys.path.insert(0,'/holdout_task')
import batch_solver as base
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=5,suppress=True,linewidth=160)
def fit(g,y): return np.linalg.lstsq(g,y,rcond=None)[0]
def design(*x): return np.column_stack((np.ones(len(x[0])),*x))
def profile(u,v,phi,bounds):
 def err(a):
  g=phi(u,a);return np.sum((g@fit(g,v)-v)**2)
 r=minimize_scalar(err,bounds=bounds,method='bounded',options={'xatol':1e-12});g=phi(u,r.x);return r.x,fit(g,v),np.sqrt(r.fun/v.size)
def operators(s,kind='sqrt_rat'):
 b=s['panel_b'];g=design(b['u']); bb=fit(g,b['v']); pan=s['panel_c']; u,v,w=[pan[k] for k in ('u','v1','v2')];
 odd=lambda x,a:design(np.sign(x)*(-np.expm1(-a*abs(x))))
 even={'sqrt_rat':lambda x,a:design(x,x*x/np.sqrt(1+a*x*x)), 'sqrt':lambda x,a:design(x,np.sqrt(1+a*x*x)-1), 'quad':lambda x,a:design(x,x*x)}[kind]
 a,bc,e=profile(u,v,odd,(.1,5));q,bd,h=profile(u,w,even,(.0001,2));return (lambda x:design(x)@bb),(lambda x:odd(x,a)@bc),(lambda x:even(x,q)@bd),(a,e,q,h)
x=s['train']['x'];y=s['train']['y'];z=s['train']['traj'];Y=np.c_[y,z]; n=len(x); i=np.arange(n); rng=np.random.RandomState(228); radial=np.argsort(np.linalg.norm(x,axis=1))[-15:];folds=[(i[:52],i[52:])]+[(np.setdiff1d(i,j),j) for j in np.array_split(rng.permutation(n),5)]+[(np.setdiff1d(i,radial),radial)]
def assess(g,lam=.0003, target=Y, inv=lambda p:p):
 ee=[]
 for tr,va in folds:
  p=np.repeat(lam,g.shape[1]);p[0]=0; b=np.linalg.solve(g[tr].T@g[tr]+np.diag(p),g[tr].T@target[tr]); e=np.abs(inv(g[va]@b)-Y[va]);ee.append([e[:,0].mean(),e[:,1:].mean(),e[:,-1].mean()]);
 e=np.array(ee);return np.array([e[0],e[1:6].mean(0),e[1:6].max(0),e[-1]])
if __name__=='__main__':
 for k in ['panel_a','panel_b']:
  d=s[k];g=design(d['u']);e=np.linalg.norm(d['v']-g@fit(g,d['v']),axis=1);print(k,np.sort(e),flush=True)
 print('range y, z, times, summary',y.min(),y.max(),z.min(),z.max(),s['eval_times'],[y.mean(),y.std(),z[:,-1].mean(),z[:,-1].std()]);print('baseline metrics',assess(base._features(s,x)))
 for k in ('sqrt_rat','sqrt','quad'):
  B,C,D,p=operators(s,k);f=B(x); c=C(f);d=D(f);m=(f[:,:,None]*c[:,None,:]).reshape(n,16);g=design(f,c,d,m);print(k,p,assess(g))
 B,C,D,p=operators(s);f=B(x);c=C(f);d=D(f);m=(f[:,:,None]*c[:,None,:]).reshape(n,16);g=design(f,c,d,m); coef=fit(g[:52],Y[:52]); print('coef y, state first, last',coef[:,[0,1,-1]]); e=y-g@coef[:,0];print('y training residual rms',np.std(e[:52]),'corr y/pred',np.corrcoef([e,y,g@coef[:,0]])); print('radial train test',np.sort(np.linalg.norm(x,axis=1))[-17:],np.sort(np.linalg.norm(s['test']['x'],axis=1))[-12:]);
 for width in (1.05,1.2,1.5,2,3,5):
  if width<=max(abs(y)):continue
  T=Y.copy();T[:,0]=np.arctanh(y/width)*width
  def inv(p):p[:,0]=width*np.tanh(p[:,0]/width);return p
  print('link tanh',width,assess(g,.0003,T,inv)[:,0]);
 for width in (max(abs(y))*1.02,1.3,1.7,2,3):
  T=Y.copy();T[:,0]=np.arcsin(y/width)*width
  def inv(p):p[:,0]=width*np.sin(p[:,0]/width);return p
  print('link sine',width,assess(g,.0003,T,inv)[:,0]);
