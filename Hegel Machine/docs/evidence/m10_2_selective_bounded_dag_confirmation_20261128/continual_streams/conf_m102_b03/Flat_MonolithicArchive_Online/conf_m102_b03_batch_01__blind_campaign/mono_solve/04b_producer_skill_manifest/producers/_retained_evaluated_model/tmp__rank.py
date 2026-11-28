from operators import *
from sklearn.preprocessing import PolynomialFeatures
from scipy.linalg import qr
from scipy.optimize import least_squares
from itertools import combinations
x=s['train']['x']; z=(x-a)/b; Y=np.c_[s['train']['y'],s['train']['traj']]; t=s['eval_times'];print('tel summary',s['telemetry_vec'],s['train']['summary']);print('X range',z.min(0),z.max(0)); print('Ysv',np.linalg.svd(Y,compute_uv=False));
def th(r):return -np.expm1(-t[:,None]*r)
q=least_squares(lambda r:(th(r)@fit(th(r),Y[:,1:].T)-Y[:,1:].T).ravel(),[.38,1.8,5.4]); H=th(q.x); T=fit(H,Y[:,1:].T).T; print('rates',q.x,'Y and A corr',np.corrcoef(np.c_[Y[:,0],T].T)); print('y~A',fit(F(T),Y[:,0]),abs(F(T)@fit(F(T),Y[:,0])-Y[:,0]).mean()); print('first rows',np.c_[z,Y[:,0],T][:20]);
for w in [z,Bf(z),Binv(z)]:
 print('coord',w[0]);
 for dim in [1,2,3]:
  P=PolynomialFeatures(dim).fit_transform(w); R=np.c_[Y[:,0],T]-P@fit(P,np.c_[Y[:,0],T]); print(dim,abs(R).mean(0),np.corrcoef(R.T));
# Greedy shared support in a hierarchy of compact, scale-tested basis functions.
from scipy.optimize import minimize_scalar
c=s['panel_c']; u=c['u']; pairs=list(combinations(range(4),2)); cross=lambda v:np.column_stack([v[:,i]*v[:,j] for i,j in pairs]); C=fit(F(cross(u)),c['v1']);C[abs(C)<.0005]=0; D=fit(F(np.cos(2.4*u)),c['v2']);
f=lambda w:F(cross(w))@C; h=lambda w:F(np.cos(2.4*w))@D
R=[]; labels=[]
def add(v,k):
 if v.ndim==1:v=v[:,None]
 R.append(v);labels.extend([(k,i) for i in range(v.shape[1])])
add(np.ones(len(z)),'bias')
for k,w in [('z',z),('Bz',Bf(z)),('Bt',Binv(z)),('BL',z@B[1:])]:
 poly=PolynomialFeatures(4,include_bias=False); add(poly.fit_transform(w),k+'poly')
 X={'w':w,'p':f(w),'c':h(w),'sum':f(w)+h(w),'diff':f(w)-h(w),'bf':Bf(f(w)),'bc':Bf(h(w)),'pofc':f(h(w)),'cofp':h(f(w))}
 for j,(m,v) in enumerate(X.items()):
  add(v,(k,m)); add(v*v,(k,m,'sq'));add(v*v*v,(k,m,'cub'))
  for g in [.22,.5,1,1.8,2.4,3.2]:
   for nm,fn in [('sin',np.sin),('cos',np.cos),('tanh',np.tanh),('rat',lambda a:a/(1+a*a)),('gauss',lambda a:np.exp(-a*a))]:add(fn(v*g),(k,m,nm,g))
  for l,q in X.items():
   add(v*q,(k,m,l,'prod'))
   for g in [.22,.5,1,2]:
    add(v/(1+g*q*q),(k,m,l,'rat',g));add(v*np.tanh(g*q),(k,m,l,'tg',g));add(v/(1+np.exp(-np.clip(g*q,-60,60))),(k,m,l,'sig',g))
M=np.column_stack(R); Yr=np.c_[Y[:,0],T];
for sel in [[0],[1],[2],[3],[0,1,2,3]]:
 for n in [40,52]:
  target=Yr[:,sel]; target=target/target[:n].std(0); norm=np.linalg.norm(M[:n],axis=0); norm[norm<1e-9]=1; A=M/norm; ids=[0]; r=target[:n]-target[:n].mean(0);P=A[:n];
  for i in range(20):
   j=np.argmax(np.linalg.norm(P.T@r,axis=1));ids.append(j); coef=fit(P[:,ids],target[:n]); e=target-A[:,ids]@coef;r=e[:n];
   if i in [2,5,8,11,15,19]:print('omp',sel,n,i,np.mean(abs(r),0),np.mean(abs(e[n:]),0),[labels[k] for k in ids[-4:]])
