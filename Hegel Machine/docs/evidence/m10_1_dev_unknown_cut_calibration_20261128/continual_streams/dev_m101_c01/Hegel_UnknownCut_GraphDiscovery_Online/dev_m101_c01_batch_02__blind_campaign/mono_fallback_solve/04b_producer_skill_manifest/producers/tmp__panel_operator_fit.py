from structure import *
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
C=s['panel_c']; u=C['u']; vv=C['v1']; ww=C['v2']
forms={
 'v1_tanh':(vv,lambda u,p:np.c_[np.ones(len(u)),np.tanh(p[0]*u)], [1.6]),
 'v1_tanh_lin':(vv,lambda u,p:np.c_[np.ones(len(u)),u,np.tanh(p[0]*u)], [2.2]),
 'v1_tanh_even':(vv,lambda u,p:np.c_[np.ones(len(u)),u,u*u,np.tanh(p[0]*u)], [2.2]),
 'v1_tanh_sin':(vv,lambda u,p:np.c_[np.ones(len(u)),np.sin(p[1]*u),np.tanh(p[0]*u)], [1.7,1.]),
 'v1_tanh_shift':(vv,lambda u,p:np.c_[np.ones(len(u)),u,np.tanh(p[0]*u+p[1:])], [2.2,0,0,0,0]),
 'v2_sq':(ww,lambda u,p:np.c_[np.ones(len(u)),u,u*u], []),
 'v2_sq_tanh':(ww,lambda u,p:np.c_[np.ones(len(u)),u*u,np.tanh(p[0]*u)], [.3]),
 'v2_sq_sin':(ww,lambda u,p:np.c_[np.ones(len(u)),u*u,np.sin(p[0]*u)], [.3]),
 'v2_cos':(ww,lambda u,p:np.c_[np.ones(len(u)),u,np.cos(p[0]*u)], [.3]),
 'v2_cubic':(ww,lambda u,p:np.c_[np.ones(len(u)),u,u*u,u**3], []),
 'v2_joint':(ww,lambda u,p:np.c_[np.ones(len(u)),u,np.cos(p[0]*u),np.sin(p[1]*u)], [.4,.4]),
}
def fit(k,i):
 y,fn,p=forms[k]
 def residual(p):
  F=fn(u[i],p); return (F@np.linalg.lstsq(F,y[i],rcond=None)[0]-y[i]).ravel()
 if p:p=least_squares(residual,p,max_nfev=300).x
 w=np.linalg.lstsq(fn(u[i],p),y[i],rcond=None)[0]
 return p,lambda U:fn(U,p)@w,np.mean(abs(residual(p)))

def kfit(K,L,Y):
 I=np.eye(len(Y)); K=np.concatenate([K,[I]]); L=np.concatenate([L,[np.zeros((L.shape[1],L.shape[2]))]]); scale=np.mean(np.diagonal(K,axis1=1,axis2=2),axis=1); K=K/scale[:,None,None]; L=L/scale[:,None,None]; Y=np.asarray(Y); d=np.std(Y,axis=0); Y=Y/d; m=Y.shape[1];
 def ml(p):
  w=np.exp(p); H=cho_factor(np.einsum('k,kij->ij',w,K),check_finite=False); a=cho_solve(H,Y,check_finite=False); inv=cho_solve(H,I,check_finite=False); return (a*Y).sum()/2+m*np.log(np.diag(H[0])).sum(),w*np.einsum('kij,ij->k',K,m*inv-a@a.T)/2
 p=minimize(ml,np.r_[np.zeros(len(K)-1),-8.],jac=True,bounds=[(-19,8)]*(len(K)-1)+[(-18,-1)],method='L-BFGS-B').x; w=np.exp(p); return (np.einsum('k,kij->ij',w,L)@np.linalg.solve(np.einsum('k,kij->ij',w,K),Y))*d

if __name__=='__main__':
 for k,(truth,fn,start) in forms.items():
  errs=[]
  for ix in np.array_split(np.random.default_rng(813).permutation(24),4):
   p,f,e=fit(k,np.setdiff1d(np.arange(24),ix)); errs.append(np.mean(abs(f(u[ix])-truth[ix])))
  p,f,e=fit(k,np.arange(24)); print(k,np.round(p,3),round(e,6),np.round(errs,5))
 for pn in ('panel_b',):
  pa=s[pn]; print(pn,'affine max residual',np.max(abs(apply(pa['u'],affine(pa))-pa['v'])))
 v=fit('v1_tanh_lin',np.arange(24))[1]; w=fit('v2_cubic',np.arange(24))[1]; ids=np.arange(72); rng=np.random.default_rng(511); order=np.argsort(np.linalg.norm(x,axis=1)); splits=[(ids[:56],ids[56:])]+[(i[:56],i[56:]) for i in [rng.permutation(ids) for _ in range(4)]]+[(order[:56],order[56:])]; Y=np.c_[y,z]; P=poly(xa); D=[P@P.T,np.exp(-cdist(xa,xa,'sqeuclidean')/8)]
 for name,co in [('X',x),('A',xa),('B',xb),('AB',ab),('BA',ba)]:
  V,W=v(co),w(co); F=[np.c_[V,W],poly(V),poly(W), (V[:,:,None]*W[:,None,:]).reshape(len(x),16),np.exp(-cdist(V,V,'sqeuclidean')/8)]
  blocks=D+[f@f.T for f in F[:-1]]+[F[-1]]; blocks=np.array(blocks); errs=[]
  for i,j in splits:
   K,L=blocks[:,i[:,None],i],blocks[:,j[:,None],i]; e=abs(np.c_[kfit(K,L,Y[i,:1]),kfit(K,L,Y[i,1:])]-Y[j]); errs.append([e[:,0].mean(),e[:,1:].mean()]);
  print('C composition',name,np.round(errs,5).T)
