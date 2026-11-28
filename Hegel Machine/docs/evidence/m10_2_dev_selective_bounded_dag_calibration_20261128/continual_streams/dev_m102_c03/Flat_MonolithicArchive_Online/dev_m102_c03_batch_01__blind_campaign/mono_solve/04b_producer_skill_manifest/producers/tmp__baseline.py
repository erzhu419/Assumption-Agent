import pickle,numpy as np
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
from sklearn.preprocessing import PolynomialFeatures
from itertools import combinations
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; p=s['panel_a']; u=p['u']; v=p['v']; U=np.c_[np.ones(len(u)),u]; keep=np.ones(len(u),bool)
for _ in range(5):
 a=np.linalg.lstsq(U[keep],v[keep],rcond=None)[0]; e=np.linalg.norm(v-U@a,axis=1); keep=e<max(.08,3*np.median(e))
np.set_printoptions(precision=5,suppress=True); print('panel A',keep.sum(),a, 'res',np.sort(e));
x=s['train']['x']; z=(x-a[0])@np.linalg.inv(a[1:]); t=s['eval_times']; tr=s['train']; n=52; y=tr['y']; Y=tr['traj']; q=-np.expm1(-t[:,None]*np.array([.4,1.82])); A=np.linalg.lstsq(q,Y.T,rcond=None)[0].T; T=np.c_[y,A]; print('times',t); print('A corr',np.corrcoef(T.T)); print('z ranges',z.min(0),z.max(0))
for pk,k in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 p=s[pk]; print(pk,k)
 for d in (1,2):
  U=PolynomialFeatures(d).fit_transform(p['u']); a=np.linalg.lstsq(U,p[k],rcond=None)[0]; r=p[k]-U@a; loo=r/(1-np.sum(np.linalg.qr(U)[0]**2,axis=1))[:,None]; print('deg',d,'fit',np.sqrt(np.mean(r*r,axis=0)),'loo',np.sqrt(np.mean(loo*loo,axis=0))); print(np.round(a[np.argsort(np.max(abs(a),axis=1))[-8:]],3))
R=[]
def add(name,K,ls):
 for l in ls:
  P=K[n:,:n]@np.linalg.solve(K[:n,:n]+l*np.eye(n),T[:n]); e=np.mean(abs(P-T[n:]),axis=0); h=np.mean(abs((P[:,1:]-T[n:,1:])@q.T)); R.append((name,l,*e,h))
for zz in (x,z):
 for d in range(1,6):
  F=PolynomialFeatures(d).fit_transform(zz); add('p'+str(d),F@F.T,[1e-6,.001,.01,.1,1.])
 for k in [.25,.5,1,2,4,8]:
  d=cdist(zz,zz); add('rbf'+str(k),np.exp(-(d/k)**2),[1e-6,.001,.01,.1]); h=np.sqrt(5)*d/k; add('matern'+str(k),(1+h+h*h/3)*np.exp(-h),[1e-6,.001,.01,.1])
print('ranked y',sorted(R,key=lambda e:e[2])[:10]); print('ranked traj',sorted(R,key=lambda e:e[-1])[:10]); print('poly coefficient patterns:')
p=PolynomialFeatures(2); F=p.fit_transform(z); a=np.linalg.lstsq(F,T,rcond=None)[0]; print(list(zip(p.get_feature_names_out(),a))); print('poly residual stdev',np.std(T-F@a,axis=0)); print('y vs A quadratic res',np.std(y-PolynomialFeatures(2).fit_transform(A)@np.linalg.lstsq(PolynomialFeatures(2).fit_transform(A),y,rcond=None)[0]))
