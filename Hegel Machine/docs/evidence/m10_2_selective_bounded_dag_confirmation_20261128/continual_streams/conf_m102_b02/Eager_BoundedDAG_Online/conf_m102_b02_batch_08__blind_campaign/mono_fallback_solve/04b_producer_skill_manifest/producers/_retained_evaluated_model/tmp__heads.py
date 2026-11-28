from features import *
from scipy.special import erf
from scipy.optimize import minimize_scalar
from itertools import product
from scipy.linalg import qr
rng=np.random.default_rng(776)
F=np.c_[z,b,c,e]; Z=D(np.c_[z,b]); i,j=splits[0]
for typ in ['grow','decay','drift','decay_drift','two']:
 def phi(r):
  v=np.exp(-r*ts)
  return dict(grow=(1-v)[:,None],decay=D(v[:,None]),drift=np.c_[1-v,ts],decay_drift=np.c_[np.ones(len(ts)),v,ts],two=np.c_[1-v,1-np.exp(-.25*ts)])[typ]
 vals=[]
 for row in g:
  def loss(r):
   a=phi(r);return np.mean((a@fit(a,row)-row)**2)
  r=minimize_scalar(loss,bounds=(.03,4),method='bounded').x;a=phi(r);vals.append(np.r_[r,fit(a,row),loss(r)])
 vals=np.array(vals); print('traj',typ,'rms',np.sqrt(vals[:,-1].mean()),'param quantiles',np.quantile(vals[:,:-1],[0,.1,.5,.9,1],axis=0).round(4)); print('affine params',np.mean(abs(Z[j]@fit(Z[i],vals[i,:-1])-vals[j,:-1]),axis=0));print('first',vals[:6,:-1].round(4))
print('telemetry',s['telemetry_vec'],'summary',t['summary']);print('y+features',np.c_[y,z,b][:12].round(3))
fn={'tanh':(np.tanh,lambda q:1-np.tanh(q)**2),'sin':(np.sin,np.cos),'asinh':(np.arcsinh,lambda q:1/np.sqrt(1+q*q)),'erf':(erf,lambda q:2/np.sqrt(np.pi)*np.exp(-q*q)),'rat':(lambda q:q/(1+q*q),lambda q:(1-q*q)/(1+q*q)**2),'cos':(np.cos,lambda q:-np.sin(q))}
for cols,N in [('zb',Z),('all',D(F))]:
 for name,(f,df) in fn.items():
  for nh in [1,2,3]:
   d=N.shape[1]; X=N[i]; I=np.eye(d*(nh+1)+nh); best=None
   for strength in [.002,.025]:
    for seed in range(5):
     def pred(q,X):
      p=q[:d]; W=q[d:-nh].reshape(d,nh); a=q[-nh:];return X@p+f(X@W)@a
     def loss(q):return np.r_[pred(q,X)-y[i],strength*q]
     def jac(q):
      W=q[d:-nh].reshape(d,nh);a=q[-nh:]; P=X@W
      return np.r_[np.c_[X,(X[:,:,None]*(df(P)*a)[:,None,:]).reshape(len(X),-1),f(P)],strength*I]
     st=np.r_[fit(X,y[i]),rng.normal(0,.6,d*nh),rng.normal(0,.5,nh)]
     R=least_squares(loss,st,jac=jac,max_nfev=500)
     if best is None or R.cost<best.cost:best=R
    print(cols,name,nh,strength,round(np.mean(abs(pred(best.x,N[j])-y[j])),5),round(np.std(pred(best.x,X)-y[i]),5),flush=True)
    if nh==1 and np.mean(abs(pred(best.x,N[j])-y[j]))<.07:print(best.x.round(4))
    best=None
