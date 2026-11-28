import numpy as np,pickle
from scipy.optimize import least_squares
from sklearn.preprocessing import PolynomialFeatures
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; d=s['train']; x=d['x']; t=s['eval_times']; T=d['traj']; y=d['y']; np.set_printoptions(precision=4,suppress=True,linewidth=140)
def lin(x,y):return np.linalg.lstsq(x,y,rcond=1e-10)[0]
def one(x):return np.c_[np.ones(len(x)),x]
def poly(x,k):return PolynomialFeatures(k).fit_transform(x)
def basis(r):return -np.expm1(-t[:,None]*r)
r=least_squares(lambda r:(basis(r)@lin(basis(r),T[:52].T)-T[:52].T).ravel(),[.4,1.8],bounds=([.01,.7],[.8,7]),gtol=1e-13).x
A=lin(basis(r),T.T).T; Y=np.c_[y,A]; print('rates',r)
p=s['panel_a']; u,v=p['u'],p['v']; keep=np.ones(len(u),bool)
for _ in range(5):
 f=one(u); e=np.linalg.norm(f@lin(f[keep],v[keep])-v,axis=1); keep=e<4*np.median(e)
print('a keep',sum(keep))
for j in range(4):
 e=v[:,j]-one(u)@lin(one(u)[keep],v[keep,j]); D=poly(u,3); b=lin(D[keep],e[keep]); names=PolynomialFeatures(3).fit(u).get_feature_names_out(); ix=np.argsort(-abs(b))[:9];print(j, list(zip(names[ix],b[ix])),np.std((D@b-e)[keep])); print(np.c_[u,v[:,j],e][keep][:8])
for name in ['panel_a','panel_c']:
 p=s[name]; u=p['u']; keep0=keep if name=='panel_a' else np.ones(len(u),bool)
 for k,v in p.items():
  if k=='u':continue
  features={'lin':one(u),'quad':poly(u,2),'products':one(u[:,[0,0,1,2]]*u[:,[1,3,2,3]]),'squares':one(u*u),'sincos':one(np.c_[np.sin(u),np.cos(u)]),'tanh':one(np.tanh(u))}
  for fun,init in [(lambda w,q:w/(1+q*abs(w)),.85),(lambda w,q:w*w/(1+q*w*w),.3),(lambda w,q:np.sin(q*w),1.),(lambda w,q:np.cos(q*w),1.),(lambda w,q:np.tanh(q*w),1.)]:
   for mode in range(3):
    def design(q):
     a=fun(u,q[0]);return one(a) if mode==0 else one(np.c_[u,a]) if mode==1 else one(np.c_[u,u*u,a]);
    def err(q):
     a=design(q)[keep0]; z=v[keep0];return (a@lin(a,z)-z).ravel()
    q=least_squares(err,[init],bounds=(.04,6)).x; features[f'{init} {fun.__code__.co_firstlineno} {mode} {q[0]:.5f} {str(fun.__code__.co_names)}']=design(q)
  print(name,k,sorted((round(np.std((F@lin(F[keep0],v[keep0])-v)[keep0]),7),fn) for fn,F in features.items())[:16]);
