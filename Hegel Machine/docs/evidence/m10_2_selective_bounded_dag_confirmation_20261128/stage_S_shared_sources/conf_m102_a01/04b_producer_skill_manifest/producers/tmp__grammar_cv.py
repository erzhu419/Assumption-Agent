import pickle,numpy as np
from scipy.optimize import least_squares
from scipy.special import erf
from sklearn.preprocessing import PolynomialFeatures
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; X=s['train']['x']; T=s['train']['traj']; t=s['eval_times']; y=s['train']['y']; n=len(X)
aug=lambda z:np.c_[np.ones(len(z)),z]
a=s['panel_a']; z=a['u']; v=a['v']; A=[]
for i in range(4):
 D=aug(z[:,i]); w=np.linalg.lstsq(D,v[:,i],rcond=None)[0];
 for _ in range(6):
  e=abs(D@w-v[:,i]); m=e<max(.0005,3*np.median(e)); w=np.linalg.lstsq(D[m],v[m,i],rcond=None)[0]
 A.append(w)
A=np.array(A); b=s['panel_b']; B=np.linalg.lstsq(aug(b['u']),b['v'],rcond=None)[0]; print('A,B',A,B,'Bsv',np.linalg.svd(B[1:],compute_uv=False)); print('Bres',np.std(aug(b['u'])@B-b['v'],0)); ai=lambda z:(z-A[:,0])/A[:,1]; aa=lambda z:z*A[:,1]+A[:,0]; bb=lambda z:aug(z)@B; bi=lambda z:(z-B[0])@np.linalg.inv(B[1:]);
E=lambda k:-np.expm1(-np.outer(t,k)); r=least_squares(lambda k:(T-T@np.linalg.pinv(E(k).T)@E(k).T).ravel(),[.22,1.52]); U=np.c_[y,T@np.linalg.pinv(E(r.x).T)]; Y=np.c_[y,T];
for name,z in [('x',X),('ai',ai(X)),('aib',bb(ai(X))),('aibi',bi(ai(X)))]:
 p=PolynomialFeatures(3); F=p.fit_transform(z); W=np.linalg.lstsq(F,U,rcond=None)[0]; print(name,'fit rms',np.std(U-F@W,0));
 for col in range(3):
  print('response',col,[(p.get_feature_names_out()[j],round(W[j,col],4)) for j in np.argsort(abs(W[15:,col]))[-8:][::-1]+15]);
C=s['panel_c']; u=C['u']; v=C['v1'];
for name,fun,k,lo,hi in [ ('erf',lambda z,k:erf(k[0]*z),[1], [.01],[8]), ('atanlin',lambda z,k:np.arctan(k[0]*z)+k[1]*z,[2,0],[.01,-2],[10,2]),('smoothlin',lambda z,k:z/np.sqrt(1+k[0]*z*z)+k[1]*z,[2,.05],[.01,-2],[15,2]),('tanhlin',lambda z,k:np.tanh(k[0]*z)+k[1]*z,[2,.05],[.01,-2],[10,2]),('soft',lambda z,k:z/(1+k[0]*abs(z)**k[1])**(1/k[1]),[2,2],[.01,.4],[20,8]),('tanhrat',lambda z,k:np.tanh(k[0]*z)/np.sqrt(1+k[1]*z*z)+k[2]*z,[2,.1,.01],[.05,-.62,-2],[8,10,2]),('asinhlin',lambda z,k:np.arcsinh(k[0]*z)+k[1]*z,[2,0],[.01,-5],[20,5])]:
 def ls(k):
  D=aug(fun(u,k)); return D,np.linalg.lstsq(D,v,rcond=None)[0]
 r=least_squares(lambda k:(lambda D,W:(D@W-v).ravel())(*ls(k)),k,bounds=(lo,hi),max_nfev=200); D,W=ls(r.x); print('v1',name,r.x,np.std(r.fun),np.linalg.svd(W[1:],compute_uv=False))
R=np.random.default_rng(423); splits=[(np.arange(52),np.arange(52,72))]+[(p[:52],p[52:]) for p in [R.permutation(n) for _ in range(20)]]+[(np.sort(np.argsort(X[:,i]*sign)[:52]),np.sort(np.argsort(X[:,i]*sign)[52:])) for i in range(4) for sign in (-1,1)]
I,J=map(np.array,zip(*splits));
def cv(F,l):
 D=F[I]; P=F[J]; W=np.linalg.solve(D.transpose(0,2,1)@D+np.diag(np.broadcast_to(l,(D.shape[-1],))),D.transpose(0,2,1)@Y[I]); e=abs(P@W-Y[J]); z=np.c_[e[:,:,0].mean(1),e[:,:,1:].mean((1,2)),e[:,:,1:].mean(1).max(1)]; return np.vstack([z[0],z[1:21].mean(0),z[1:21].max(0),z[21:].mean(0),z[21:].max(0)])
results=[]
for nm,z in [('x',X),('ai',ai(X)),('b',bb(X)),('aib',bb(ai(X))),('aibi',bi(ai(X)))]:
 for d in [2,3,4,5]:
  p=PolynomialFeatures(d); F=p.fit_transform(z); deg=p.powers_.sum(1); F/=np.sqrt(np.mean(F**2,axis=0));
  for l in [.0002,.003,.025,.15,.55]:
   for mult in [1,5,25]:
    e=cv(F,np.where(deg<=1,1e-9,l*float(mult)**(deg-2))); results.append((nm,d,l,mult,e))
print('baseline',cv(PolynomialFeatures(2).fit_transform(X),1e-9))
for row in sorted(results,key=lambda row:np.mean(row[-1]*np.array([.2,1,.7,.6,.3])[:,None]))[:12]: print(row[:-1],np.round(row[-1],4).tolist())
