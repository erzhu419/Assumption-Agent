from gp import *
from scipy.linalg import svd
from sklearn.preprocessing import PolynomialFeatures
for k in range(2):
 u=s['panel_c']['u']; v=s['panel_c']['v'+str(k+1)];
 for extra in ([False,True] if k==0 else [True]):
  def basis(u,p):
   f=u/(1+p[0]*abs(u)) if k==0 else np.cos(p[0]*u)
   return one(np.c_[u,f]) if extra else one(f)
  q=least_squares(lambda p:(basis(u,p)@ls(basis(u,p),v)-v).ravel(),[.85 if k==0 else 1.27]); print('panel',k,extra,q.x,np.std(q.fun))
H=lambda p:-np.expm1(-t[:,None]*p); q=least_squares(lambda p:(T[tr]-ls(H(p),T[tr].T).T@H(p).T).ravel(),[1.8,.4]); Y=np.c_[y,ls(H(q.x),T.T).T]; C=z['xaC']; D=z['xaD']; B=z['xaB']; X=z['xa'];
P=C*D; pairs=lambda A,E:np.einsum('ni,nj->nij',A,E).reshape(len(A),-1); pol=lambda X,i:PolynomialFeatures(i,include_bias=False).fit_transform(X); F=np.c_[D,C,P];
def reg(F,lam=1e-6,detail=False):
 A=np.c_[np.ones(len(F)),(F-F[tr].mean(0))/np.maximum(F[tr].std(0),1e-10)]; u,d,v=svd(A[tr],full_matrices=False); sh=d*d/(d*d+lam); co=v.T@((sh/np.maximum(d,1e-14))[:,None]*(u.T@Y[tr])); E=Y-A@co; loo=(abs(E[tr]/np.maximum(1-u**2@sh,1e-6)[:,None])).mean(0); mae=abs(E[va]).mean(0); et=E[:,1:]@H(q.x).T; print('loo',loo,'mae',mae,'state',abs(et[va]).mean(0).max(),'train rmse',np.std(E[tr],0));
 if detail:print('coefs',np.round(co[1:]/F[tr].std(0)[:,None],4)); print('res corr with all',np.corrcoef(np.c_[E[tr],B[tr],C[tr],D[tr],Y[tr]].T)[:3,-3:]);
 return E
for nm,A in [('CDdiag',F),('CDdiag+B',np.c_[F,B]),('CDsum',np.c_[D,C,P.sum(1)]),('C vs D',np.c_[D,C,pairs(C,D)]),('CDdiag+quad B',np.c_[F,pol(B,2)]),('CDdiag+quad D',np.c_[F,pol(D,2)[:,4:]]),('CDdiag+quad C',np.c_[F,pol(C,2)[:,4:]]),('CDdiag+quad CD',np.c_[F,pol(D,2)[:,4:],pol(C,2)[:,4:]]),('CDdiag+triple',np.c_[F,C*C*D,C*D*D,P.sum(1)[:,None]*np.c_[C,D]]),('CDdiag+pairings',np.c_[F,C*B,D*B,C*X,D*X])]:
 print(nm); reg(A,detail=nm in ('CDsum','CDdiag','CDdiag+B')); 
 if A.shape[1]>18:reg(A,.01)
# sparse, with calibrated branch linear terms retained and a broad dictionary for the remaining state residual.
base=one(np.c_[C,D,P]); Dic=np.c_[pol(np.c_[C,D],3)[:,8:],pol(B,3),C*B,D*B,np.sin(C),np.sin(D),np.tanh(C),np.tanh(D),B/(1+abs(B))]; names=np.r_[PolynomialFeatures(3,include_bias=False).fit(np.c_[C,D]).get_feature_names_out()[8:],['B'+n for n in PolynomialFeatures(3,include_bias=False).fit(B).get_feature_names_out()], [f'{nm}{i}' for nm in ['C*B','D*B','sinC','sinD','tanhC','tanhD','softB'] for i in range(4)]];
for j in [1,2]:
 sel=[]
 for step in range(12):
  U=np.c_[base,Dic[:,sel]]; Q=svd(U[tr],full_matrices=False)[0]; e=Y[:,j]-U@ls(U[tr],Y[tr,j]); G=Dic[tr]-Q@(Q.T@Dic[tr]); n=np.linalg.norm(G,axis=0); sc=np.where(n>1e-8,abs(G.T@e[tr])/np.maximum(n,1e-9),0); a=sc.argmax(); sel.append(a); print('sparse',j,step,round(abs(e[va]).mean(),5),round(np.std(e[tr]),5),names[a]);
