exec(open('/tmp/panels07.py').read().split('for label,q')[0])
from itertools import combinations
from scipy.optimize import minimize_scalar

def aff(x):return np.column_stack((np.ones(len(x)),x))
def pair(x):return np.column_stack([x[:,i]*x[:,j] for i,j in ((0,1),(0,3),(1,2),(2,3))])
P=lambda x: aff(pair(x))@fit(aff(pair(c['u'])),c['v1'])
Q=lambda x: poly(x,2)@fit(poly(c['u'],2),c['v2'])
B=lambda x: poly(x,2)@fit(poly(b['u'],2),b['v'])
z=np.column_stack((y,T)); inds=list(range(0,72,18)); parts=[np.arange(52,72)]+[np.arange(i,i+18) for i in inds]
def score(x,non=None):
 ans=[]
 for h in parts:
  tr=np.setdiff1d(np.arange(n),h)
  if non:
   def feat(g):return np.column_stack((x,non(g*x[:,1:])))
   def obj(g):
    A=feat(g)[tr];return np.mean((A@fit(A,z[tr])-z[tr])**2)
   X=feat(minimize_scalar(obj,bounds=(0.2,12),method='bounded').x)
  else:X=x
  e=abs(X[h]@fit(X[tr],z[tr])-z[h]);ans.append([e[:,0].mean(),e[:,1:].mean()]);
 return np.round(ans[0],4),np.round(np.max(ans,axis=0),4)
for name,x in [('raw',raw),('u',u),('w',w),('b2',B(u))]:
 for k, fs in [('pair',pair(x)),('p',P(x)),('q',Q(x)),('pq',np.c_[P(x),Q(x)]),('pqu',np.c_[P(x),Q(x),u]),('pqw',np.c_[P(x),Q(x),w])]:
  print(name,k,score(aff(fs)),score(aff(fs),np.arctan),score(aff(fs),np.tanh))
print('traj ratio',np.round((T/np.where(abs(T[:,-1:])>.2,T[:,-1:],np.nan))[np.abs(T[:,-1])>.2][:16],4))
