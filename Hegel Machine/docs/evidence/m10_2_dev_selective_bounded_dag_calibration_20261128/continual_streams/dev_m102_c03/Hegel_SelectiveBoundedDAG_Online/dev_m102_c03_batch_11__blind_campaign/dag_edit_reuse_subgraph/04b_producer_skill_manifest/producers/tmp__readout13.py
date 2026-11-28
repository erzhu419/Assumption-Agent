import sys,pickle,numpy as np
from itertools import combinations,combinations_with_replacement
from scipy.optimize import minimize_scalar,least_squares
from scipy.linalg import orthogonal_procrustes
sys.path.insert(0,'/holdout_task/_frozen_nodes'); import node_00
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; t=s['eval_times']; x=tr['x']; q=-np.expm1(-t[:,None]*[.4127,1.8132]); y=np.c_[tr['y'],tr['traj']@np.linalg.pinv(q.T)]
def fit(h,y):return np.linalg.lstsq(h,y,rcond=None)[0]
def one(x):return np.c_[np.ones(len(x)),x]
def quad(x):return np.column_stack([x[:,i]*x[:,j] for i,j in combinations_with_replacement(range(x.shape[1]),2)])
def aq(u,v,keep):
 def des(z,w,j):
  a,b=z[:,j],z[:,(j+2)%4]; return np.c_[np.ones(len(z)),a,b,np.tanh(w[0]*a),np.tanh(w[1]*b)]
 def coeff(w): return np.stack([fit(des(u[keep],w,j),v[keep,j]) for j in range(4)])
 def res(w):return np.column_stack([des(u[keep],w,j)@c for j,c in enumerate(coeff(w))])-v[keep]
 w=least_squares(lambda w:res(w).ravel(),[1.1,1.1],bounds=(.25,4)).x; c=coeff(w); print('a',w,c,np.sqrt(np.mean(res(w)**2)))
 def invert(x):
  z=(x-c[:,0])/1.5
  for _ in range(30):
   a,b=z,np.roll(z,2,axis=1); ta,tb=np.tanh(w[0]*a),np.tanh(w[1]*b); r=c[:,0]+c[:,1]*a+c[:,2]*b+c[:,3]*ta+c[:,4]*tb-x; J=np.zeros((len(z),4,4)); j=np.arange(4); J[:,j,j]=c[:,1]+c[:,3]*w[0]*(1-ta*ta); J[:,j,(j+2)%4]=c[:,2]+c[:,4]*w[1]*(1-tb*tb); z-=np.clip(np.linalg.solve(J,r[...,None])[...,0],-.35,.35)
  return z
 return invert
u,v=(s['panel_a'][k] for k in ('u','v')); keep=np.argsort(np.linalg.norm(v-one(u)@fit(one(u),v),axis=1))[:45]; z=aq(u,v,keep)(x); print('z ranges',z.min(0),z.max(0)); u=s['panel_c']['u']; v1=s['panel_c']['v1']; v2=s['panel_c']['v2'];
for fname,f in [('tanh',np.tanh),('sin',np.sin),('atan',np.arctan),('softsign',lambda a:a/(1+abs(a))),('gauss',lambda a:np.expm1(-a*a))]:
 for lin in (False,True):
  def des(a,w):return np.c_[one(a),f(w*a)] if lin else one(f(w*a))
  def cost(w):
   h=des(u,w);return np.mean((h@fit(h,v1)-v1)**2)
  o=minimize_scalar(cost,bounds=(.1,4),method='bounded'); print('c1',fname,lin,o.x,o.fun**.5)
print('study x ranges',x.min(0),x.max(0));
w=z-1.837*np.expm1(-.2922*z*z); g=np.tanh(1.59*z); b=one(z)@fit(one(s['panel_b']['u']),s['panel_b']['v']);
for key,h in [('raw',one(x)),('z',one(z)),('w',one(w)),('g',one(g)),('b',one(b)),('zg',np.c_[one(z),g]),('wg',np.c_[one(w),g]),('quad-z',np.c_[one(z),quad(z)]),('quad-w',np.c_[one(w),quad(w)]),('quad-g',np.c_[one(g),quad(g)]),('quad-wg',np.c_[one(w),g,quad(w),quad(g)]),('full',np.c_[one(w),g,quad(np.c_[w,g])])]:
 print(key,'train',np.mean(abs(h@fit(h,y)-y),0),'val',np.mean(abs(h[52:]@fit(h[:52],y[:52])-y[52:]),0))
