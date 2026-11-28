import numpy as np,pickle
from scipy.linalg import lstsq
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=5,suppress=True,linewidth=140)
def fit(D,Y):return lstsq(D,Y)[0]
def A(x):return np.c_[np.ones(len(x)),x]
def poly(x,d):
 from itertools import combinations_with_replacement
 return np.column_stack([np.ones(len(x))]+[x[:,j].prod(1) for k in range(1,d+1) for j in combinations_with_replacement(range(4),k)])
u,v=s['panel_a'].values(); print('distance to identity sorted',np.sort(np.linalg.norm(v-u,axis=1))); good=np.linalg.norm(v-u,axis=1)<.65; print('inliers',sum(good));
for k in range(4):
 a=u[:,k]; b=u[:,(k+1)%4]; D=np.c_[np.ones(len(u)),a,b,a*b,a**3]; C=fit(D[good],v[good,k]); print('A cyc',k,C,np.std((v[:,k]-D@C)[good]));
for p in ['panel_a','panel_b','panel_c']:
 u=s[p]['u']; g=good if p=='panel_a' else np.ones(len(u),bool)
 for key,v in s[p].items():
  if key=='u':continue
  for name,D in [('aff',A(u)),('quad',poly(u,2)),('ind cubic',np.c_[A(u),u**2,u**3]),('quad+ind cube',np.c_[poly(u,2),u**3]),('sin',A(np.sin(u))),('tanh',A(np.tanh(u)))]:
   D=D[g];Y=v[g]; C=fit(D,Y); r=Y-D@C; h=np.sum(np.linalg.qr(D,mode='reduced')[0]**2,axis=1); print(p,key,name,'rms',np.mean(r**2)**.5,'loo',np.mean(np.abs(r/(1-h[:,None]))))
