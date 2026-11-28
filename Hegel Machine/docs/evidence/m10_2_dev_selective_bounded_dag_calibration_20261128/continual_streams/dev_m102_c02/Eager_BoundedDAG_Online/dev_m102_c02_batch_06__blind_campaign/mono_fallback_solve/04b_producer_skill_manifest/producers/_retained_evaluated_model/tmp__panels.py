import pickle,numpy as np
from scipy.optimize import minimize_scalar,least_squares
from itertools import combinations
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=3,suppress=True,linewidth=145)
L=lambda z,y:np.linalg.lstsq(z,y,rcond=None)[0]
F=lambda x:np.c_[np.ones(len(x)),x]
P=lambda x:np.column_stack([x[:,i]*x[:,j] for i,j in combinations(range(4),2)])
def diagnose(u,v):
 R=v-F(u)@L(F(u),v)
 dic={'aff':u,'squares':np.c_[u,u*u],'pair':np.c_[u,P(u)],'quad':np.c_[u,u*u,P(u)],'abs':np.c_[u,abs(u)],'cube':np.c_[u,u**3],'u*norm':np.c_[u,u*np.sum(u*u,1)[:,None]],'u*othersq':np.c_[u,*[u*u[:,j,None]**2 for j in range(4)]], 'quadroll':np.c_[u,u*np.roll(u,1,axis=1),u*np.roll(u,2,axis=1)],'rollcube':np.c_[u,u*np.roll(u,1,axis=1)**2]}
 for name,h in [('tanh',np.tanh),('sin',np.sin),('cos',np.cos),('atan',np.arctan),('sat',lambda z:z/(1+abs(z)))]:
  def profile(k):
   z=F(np.c_[u,h(k*u)]); return np.linalg.norm(v-z@L(z,v))
  k=minimize_scalar(profile,bounds=(.1,4),method='bounded').x
  dic[name+str(round(k,3))]=np.c_[u,h(k*u)]
  dic[name+'1.25']=np.c_[u,h(1.25*u)]
 def fit(z):
  z=F(z); b=L(z,v); r=v-z@b; H=np.sum(np.linalg.qr(z,mode='reduced')[0]**2,1); return np.mean(abs(r)),np.mean(abs(r/np.maximum(1e-8,1-H[:,None]))),b
 res=sorted([(fit(z)[:2],k) for k,z in dic.items()],key=lambda a:a[0][1]); print(res[:12]);
 for k in dict.fromkeys([res[0][1],'aff','quad']):print(k,fit(dic[k])[2].T)
 # search for a common nonlinear dependency across outputs after affine term
 Z=F(u); Q=np.linalg.qr(Z)[0]; D=lambda z:z-Q@(Q.T@z)
 for f in [lambda z:z*z,np.tanh,np.sin,np.cos,lambda z:np.sqrt(1+z*z),lambda z:np.maximum(z,0),lambda z:1/(1+z*z)]:
  def features(w):return D(f(Z@w).reshape(-1,1))
  def fun(w):
   d=features(w); return (R-d@L(d,R)).ravel()
  rng=np.random.default_rng(7); best=min((least_squares(fun,rng.normal(size=5),max_nfev=100) for _ in range(12)),key=lambda a:a.cost)
  d=features(best.x); print('ridge',f.__name__,np.mean(abs(fun(best.x))),best.x,L(d,R))
for k,l in [('panel_a','v'),('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 u=s[k]['u']; v=s[k][l];
 if k=='panel_a':
  z=F(u); r=np.max(abs(v-z@L(z,v)),1); keep=r<2; u,v=u[keep],v[keep]
 print('\n',k,l);diagnose(u,v)
