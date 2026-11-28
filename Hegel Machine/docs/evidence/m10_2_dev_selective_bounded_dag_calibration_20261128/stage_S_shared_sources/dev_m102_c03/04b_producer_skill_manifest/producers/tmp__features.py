import pickle,numpy as np
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=5,suppress=True,linewidth=140)
def bias(x):return np.c_[np.ones(len(x)),x]
def fit(x,y):return np.linalg.lstsq(x,y,rcond=None)[0]
def poly(x,d):return np.column_stack([np.prod(x[:,j],axis=1) for k in range(d+1) for j in combinations_with_replacement(range(x.shape[1]),k)])
p=s['panel_a'];U=bias(p['u']);i=np.linalg.norm(p['v']-U@fit(U,p['v']),axis=1)<3; A=fit(U[i],p['v'][i]); print('A',A)
def inv(x):return (x-A[0])/np.diag(A[1:]);
p=s['panel_b'];B=fit(bias(p['u']),p['v']);
def bf(x):return bias(x)@B
p=s['panel_c']; u=p['u'];
for k in ['v1','v2']:
 v=p[k]; print(k)
 for label,z in [('raw',u),('inv',inv(u)),('b',bf(u)),('bi',bf(inv(u)))]:
  for tag,F in [('quad',poly(z,2)),('sep cubic',np.c_[bias(z),z**2,z**3]),('sep fifth',np.c_[bias(z),z**2,z**3,z**4,z**5]),('sin',np.c_[bias(z),np.sin(z),np.cos(z)]),('tanh',np.c_[bias(z),np.tanh(z),z*np.tanh(z)])]:
   w=fit(F[:19],v[:19]);print(label,tag, np.round(np.mean(abs(F[19:]@w-v[19:]),axis=0),5))
 w=fit(poly(u,2),v);print('raw quadratic coefficients',w)
 # outputwise nonlinear affine projection, with offsets before and after nonlinearity
 W=fit(bias(u),v)
 for name,fn in [('tanh',np.tanh),('sin',np.sin),('softsign',lambda a:a/(1+abs(a))),('square',lambda a:a*a),('cubic',lambda a:a+a*a*a),('num',lambda a:a/(1+a*a)),('atan',np.arctan),('exp',lambda a: np.exp(np.clip(a,-20,20)))]:
  rr=[]
  for j in range(4):
   def f(q):return q[0]+q[1]*fn(bias(u)@q[2:])
   opt=least_squares(lambda q:(f(q)-v[:,j])[:19],np.r_[0,1,W[:,j]],max_nfev=300);rr.append(np.mean(abs(f(opt.x)[19:]-v[19:,j])))
  print('ridge',name,np.round(rr,5))
x=s['train']['x']; y=s['train']['y']; T=s['train']['traj']; Y=np.c_[y,T];
for label,z in [('raw',x),('inverse',inv(x)),('affine',bias(x)@A),('b',bf(x)),('binv',bf(inv(x))),('baftera',bf(bias(x)@A))]:
 for tag,F in [('linear',bias(z)),('quad',poly(z,2)),('cubic',poly(z,3)),('sep',np.c_[bias(z),z**2,z**3,z**4]),('tanh',np.c_[bias(z),np.tanh(z),z*np.tanh(z)]),('sin',np.c_[bias(z),np.sin(z),np.cos(z)])]:
  w=fit(F[:52],Y[:52]);err=np.mean(abs(F[52:]@w-Y[52:]),axis=0);print(label,tag,err[[0,1,8]])
