from chamber_probe import *
from itertools import combinations
L=H*np.sum(theta[:2]/taus)
C=np.c_[Y,R]
labs=g['lab_id']; scale=np.r_[np.ones(len(t)),1.]
def basis(p):
 a=np.exp(p); return np.vstack([np.c_[a*-np.expm1(-t[:,None]/a),t],np.ones((1,3))])
f=least_squares(lambda p:(C-np.linalg.lstsq(basis(p),C.T,rcond=None)[0].T@basis(p).T).ravel(),np.log(taus)); print('empirical poles',np.exp(f.x)); B=np.linalg.lstsq(basis(f.x),C.T,rcond=None)[0].T; print('amplitudes [load drift fast slow linear]',np.round(np.c_[L,drift,B],4));
D=np.c_[np.ones(n),L,drift,Z, xg];print('rate models');
for inds in [(1,2),(0,1,2),(1,2,3,4),(0,1,5,6,7,8)]:
 A=D[:,inds]; b=np.linalg.lstsq(A,R,rcond=None)[0]; print(inds,b,np.round(R-A@b,4), 'mae',np.mean(abs(R-A@b)))
def fit(terms,free,m):
 A=D[:,terms]; b0=np.zeros((len(terms),3)); b0[0]=[.8,.2,0]; b0[1]=[.1,.2,.7]
 def pred(p):return A@p[2:].reshape(b0.shape)@basis(p[:2]).T
 if free:
  p=least_squares(lambda p:(pred(p)[m]-C[m]).ravel(),np.r_[np.log(taus),b0.ravel()],max_nfev=350).x
 else:
  T=basis(np.log(taus)); b=np.linalg.lstsq(np.einsum('ni,tj->ntij',A[m],T).reshape(-1,b0.size),C[m].ravel(),rcond=None)[0]; p=np.r_[np.log(taus),b]
 return p,pred(p)
for terms in [(1,2),(1,2,0),(1,2,3,4),(1,2,0,5,6,7,8)]:
 for free in (False,True):
  p,S=fit(terms,free,np.ones(n,bool)); print('terms/free/poles/coef',terms,free,np.exp(p[:2]),p[2:].reshape(-1,3)); print('LOLO curve/rate MAE',[(int(l),*np.round(abs(fit(terms,free,labs!=l)[1][labs==l]-C[labs==l]).mean(axis=0)[[0,-1]],5),round(abs(fit(terms,free,labs!=l)[1][labs==l,:-1]-Y[labs==l]).mean(),5)) for l in np.unique(labs)])
