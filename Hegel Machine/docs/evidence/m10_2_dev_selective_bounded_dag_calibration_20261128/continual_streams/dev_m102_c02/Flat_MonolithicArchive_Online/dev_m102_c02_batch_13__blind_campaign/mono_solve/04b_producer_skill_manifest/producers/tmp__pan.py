import pickle,numpy as np,itertools
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; fit=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]
def aff(x):return np.c_[np.ones(len(x)),x]
def quad(x):return np.array([x[:,i]*x[:,j] for i in range(4) for j in range(i,4)]).T
def score(h,v):
 p=np.linalg.pinv(h); e=v-h@(p@v); return np.sqrt(np.mean(e**2)),np.sqrt(np.mean((e/(1-(h*p.T).sum(1))[:,None])**2))
for p,k in [('panel_a','v'),('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 u,v=s[p]['u'],s[p][k];ok=np.linalg.norm(v,axis=1)<5;u=u[ok];v=v[ok]; print(p,k)
 C=fit(aff(u),v);print('affine',C.round(4),'rms',score(aff(u),v));print('quad',fit(np.c_[aff(u),quad(u)],v).round(4),score(np.c_[aff(u),quad(u)],v))
 for label,fn in [('cos',np.cos),('sin',np.sin),('tanh',np.tanh),('cos+sin',lambda u:np.c_[np.cos(u),np.sin(u)]),('x+cos',lambda u:np.c_[u,np.cos(u)]),('x+sin',lambda u:np.c_[u,np.sin(u)]),('x+tanh',lambda u:np.c_[u,np.tanh(u)]),('sin+x2',lambda u:np.c_[np.sin(u),u*u]),('cos+x2',lambda u:np.c_[np.cos(u),u*u]),('cos+cross',lambda u:np.c_[np.cos(u),quad(u)[:,[1,2,3,5,6,8]]])]:
  fun=lambda w:score(aff(fn(w*u)),v)[0]; fs=np.arange(.2,12,.1); i=np.argmin([fun(w) for w in fs]); w=minimize_scalar(fun,bounds=(max(.05,fs[i]-.12),fs[i]+.12),method='bounded').x; print(label,round(w,5),np.round(score(aff(fn(w*u)),v),5))
