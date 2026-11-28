import pickle,numpy as np
from scipy.optimize import least_squares
from sklearn.preprocessing import PolynomialFeatures
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; Y=np.c_[s['train']['y'],s['train']['traj']]; n=52
np.set_printoptions(precision=5,suppress=True,linewidth=140)
def aug(x):return np.c_[np.ones(len(x)),x]
def ls(x,y):return np.linalg.lstsq(x,y,rcond=None)[0]
def pol(x,d):return PolynomialFeatures(d).fit_transform(x)
def chk(X):
 p=X[n:]@ls(X[:n],Y[:n]); e=np.abs(p-Y[n:]);return np.round([e[:,0].mean(),e[:,1:].mean()],5)
for k in ['panel_a','panel_b','panel_c']:
 d=s[k]; u=d['u']; A=aug(u)
 for z,v in d.items():
  if z=='u':continue
  good=np.ones(len(u),bool); b=ls(A,v)
  if k=='panel_a':
   good=np.linalg.norm(v-A@b,axis=1)<4; b=ls(A[good],v[good]); print('A clean',b, np.sort(np.linalg.norm(v-A@b,axis=1))); af=b
  print(k,z,'ranges',v.min(0),v.max(0), 'linear',np.sqrt(np.mean((A[good]@b-v[good])**2,axis=0)))
  v=v[good];a=A[good]; uu=u[good];
  for key,f,inv in [('tanh',np.tanh,np.arctanh),('sin',np.sin,np.arcsin),('asinh',np.arcsinh,np.sinh),('softsign',lambda z:z/(1+abs(z)),lambda z:z/(1-abs(z)))]:
   e=[]; params=[]
   for j in range(v.shape[1]):
    vv=v[:,j]; w=ls(a,vv); r=least_squares(lambda p:f(a@p)-vv,w,max_nfev=1000); e.append(np.sqrt(np.mean(r.fun**2)));params.append(r.x)
   print(key,np.round(e,6))
  for deg in [2,3]:
   P=pol(uu,deg); print('poly',deg,np.sqrt(np.mean((P@ls(P,v)-v)**2,axis=0)))
  if k=='panel_b': bf=b
q=np.linalg.solve(af[1:].T,(x-af[0]).T).T; zb=aug(q)@bf; br=aug(x)@bf
print('times',s['eval_times']);print('uncentered singular',np.linalg.svd(Y[:,1:],compute_uv=False))
for key,zz in [('raw',x),('cal',q),('b',zb),('braw',br)]:
 for d in [1,2,3,4]:print(key,d,chk(pol(zz,d)))
 print(key,'linear coefficients',ls(aug(zz[:52]),Y[:52])[:,[0,1,8]])
