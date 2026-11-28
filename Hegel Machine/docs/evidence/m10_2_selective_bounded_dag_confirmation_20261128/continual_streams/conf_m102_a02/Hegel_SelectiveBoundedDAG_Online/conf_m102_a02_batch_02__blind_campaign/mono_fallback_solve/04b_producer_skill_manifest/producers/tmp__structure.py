import pickle,numpy as np
from itertools import combinations_with_replacement
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; t=s['train']['traj']
ls=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]
poly=lambda x,d:PolynomialFeatures(d).fit_transform(x)
p=s['panel_a']; u,v=p['u'],p['v']; m=np.linalg.norm(u-v,axis=1)<1
A=np.array([ls(poly(u[m,i:i+1],1),v[m,i]) for i in range(4)]); af=lambda x:x*A[:,1]+A[:,0]; ai=lambda x:(x-A[:,0])/A[:,1]
p=s['panel_b']; B=ls(poly(p['u'],1),p['v']); bf=lambda x:poly(x,1)@B
p=s['panel_c']; u=p['u']; C=[]
for key in ['v1','v2']:
 v=p[key]; print(key)
 for d in [1,2,3]:
  F=poly(u,d); es=[]
  for a,b in KFold(6,shuffle=True,random_state=11).split(u):es.extend(abs(F[b]@ls(F[a],v[a])-v[b]).ravel())
  coef=ls(F,v); print(d,'train',np.mean(abs(F@coef-v)),'CV',np.mean(es));
  if d==2: print(np.round(coef,4)); C.append(coef)
for key in ['v1','v2']:
 v=p[key]; print(key,'marginal range',v.min(0),v.max(0))

def evaluate(F):
 o=[]
 for a,b in [(np.arange(52),np.arange(52,72)),*KFold(4,shuffle=True,random_state=12).split(x)]:
  pred=F[b]@ls(F[a],np.c_[y[a],t[a]]); o.append(np.mean(abs(pred-np.c_[y[b],t[b]]),axis=0))
 o=np.array(o); return tuple(np.round([o[0,0],o[1:,0].mean(),o[1:,0].max(),o[0,1:].mean()],5))
from scipy.optimize import least_squares
from itertools import combinations_with_replacement as cwr
print('B',B)
for nm,q in [('raw',x),('A+',af(x)),('A-',ai(x)),('BA+',bf(af(x))),('BA-',bf(ai(x))),('Braw',bf(x)),('B inverse A+', (af(x)-B[0])@np.linalg.inv(B[1:])),('B inverse A-',(ai(x)-B[0])@np.linalg.inv(B[1:]))]:
 F=poly(q,3); beta=ls(F,y); print(nm,'cubic',np.round(beta[15:],4),'residual std',np.std(y-F@beta)); print('quad',np.round(beta[:15],3));
 for tag,G in [('diag3',q**3),('diag34',np.c_[q**3,q**4]),('dup3',np.stack([q[:,i]**2*q[:,j] for i in range(4) for j in range(4)],1))]: print(tag,evaluate(np.c_[F[:,:15],G]));
for key in ['v1','v2']:
 v=p[key]; aff=poly(u,1); print('C',key,'aff',ls(aff,v));
 for tag,F in [('add3',np.c_[aff,u**2,u**3]),('Q+cubes',np.c_[poly(u,2),u**3]),('Q+cubes+fours',np.c_[poly(u,2),u**3,u**4]),('add4',np.c_[aff,u**2,u**3,u**4]),('add5',np.c_[aff,u**2,u**3,u**4,u**5]),('quad_diag',np.c_[aff,u**2])]:
  b=ls(F,v); errors=[]
  for a,z in KFold(6,shuffle=True,random_state=11).split(u):errors.extend(abs(F[z]@ls(F[a],v[a])-v[z]).ravel());
  print(tag,np.mean(abs(F@b-v)),np.mean(errors));
  if tag in ('add3','quad_diag'):print(np.round(b,5))
 print('v2 relation cross',ls(np.c_[aff,p['v1'],u**2],v).T)
times=s['eval_times']; L=ls(poly(x,1),t); print('state affine',L); print('state res',np.mean(abs(t-poly(x,1)@L))); print('affine SVD',np.linalg.svd(L,compute_uv=False));
