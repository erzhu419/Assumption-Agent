import numpy as np,pickle,itertools
from scipy.optimize import least_squares
from sklearn.preprocessing import PolynomialFeatures
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=5,suppress=True,linewidth=180)
def one(x):return np.c_[np.ones(len(x)),x]
def ls(x,y):return np.linalg.lstsq(x,y,rcond=None)[0]
def affine(u,v):return ls(one(u),v)
a=s['panel_a']; ca=affine(a['u'],a['v']); keep=np.ones(len(a['u']),bool)
for _ in range(5):
 e=np.linalg.norm(a['v']-one(a['u'])@ca,axis=1); keep=e<max(.001,3*np.median(e)); ca=affine(a['u'][keep],a['v'][keep])
b=s['panel_b']; cb=affine(b['u'],b['v']); c=s['panel_c']; u=c['u']; c1=affine(u,c['v1']); poly=PolynomialFeatures(2).fit(u); cp=ls(poly.transform(u),c['v2']); print('C v2',list(zip(poly.get_feature_names_out(),cp)))
for j in range(4):
 v=c['v1'][:,j]; p=c1[:,j]; print('C1 channel',j,'u,lin,err\n',np.c_[u,one(u)@p,v-one(u)@p][:8]); best=[]
 for name,f in [('sin',np.sin),('cos',np.cos),('tanh',np.tanh)]:
  for k in range(5):
   for w in [1,2,3,5,8]:
    def pred(q):
     z=one(u)@q[:5]; inp=(z if k==4 else u[:,k]); return z+q[5]*f(q[6]*inp+q[7]);
    pfit=least_squares(lambda q:pred(q)-v,np.r_[p,.2,w,0],max_nfev=150); best.append((np.std(pfit.fun),name,k,pfit.x))
 print('best',sorted(best,key=lambda z:z[0])[:3])
t=s['eval_times']; T=s['train']['traj']; x=s['train']['x']; y=s['train']['y']; _,sig,vt=np.linalg.svd(T,full_matrices=False); print('time basis',vt[:2]);
for typ in range(5):
 def f(p):
  e=-np.expm1(-p[0]*t); g=-np.expm1(-p[1]*t)
  return [np.array([e,g]),np.array([e,t]),np.array([e,t*np.exp(-p[0]*t)]),np.array([e,t-e/p[0]]),np.array([e,np.sin(p[1]*t)])][typ]
 q=least_squares(lambda p:(T-ls(f(p).T,T.T).T@f(p)).ravel(),[2,.35],bounds=([.01,.01],[10,10])); print('time fit',typ,q.x,np.std(q.fun))
q=least_squares(lambda p:(T-ls(np.array([-np.expm1(-p[0]*t),-np.expm1(-p[1]*t)]).T,T.T).T@np.array([-np.expm1(-p[0]*t),-np.expm1(-p[1]*t)])).ravel(),[2,.35],bounds=([.01,.01],[10,10])); latent=ls(np.array([-np.expm1(-q.x[0]*t),-np.expm1(-q.x[1]*t)]).T,T.T).T; Y=np.c_[y,latent]; print('Y scale',Y.std(0))
for name,z in [('x',x),('Ax',one(x)@ca),('iAx',(x-ca[0])/np.diag(ca[1:])),('Bx',one(x)@cb),('BAx',one(one(x)@ca)@cb),('BiAx',one((x-ca[0])/np.diag(ca[1:]))@cb),('iBx',(x-cb[0])@np.linalg.inv(cb[1:]))]:
 print(name,'linear coeff',ls(one(z),Y).T)
 for freq in [.5,1,1.5,2,3,4]:
  D=one(np.c_[z,np.sin(freq*z),np.cos(freq*z),z*z]); R=np.abs(Y[52:]-D[52:]@ls(D[:52],Y[:52])); print('fourier',freq,R.mean(0))
