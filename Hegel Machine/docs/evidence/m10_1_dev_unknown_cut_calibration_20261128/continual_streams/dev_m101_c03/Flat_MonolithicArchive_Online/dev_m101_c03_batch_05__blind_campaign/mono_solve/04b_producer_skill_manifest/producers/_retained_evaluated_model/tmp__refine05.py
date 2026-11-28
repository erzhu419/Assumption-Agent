exec(open('/tmp/fit05.py').read().split('for keys in')[0])
from itertools import combinations_with_replacement
from scipy.optimize import minimize_scalar
np.set_printoptions(precision=6,suppress=True)
print('quadratic branch coefficients',fit(aff(np.c_[u,u*u,cross(u)]),v))
F=aff(u+.45*u*u); C=fit(F,v); R=v-F@C
print('v2 mixing',C,'mix singulars',np.linalg.svd(C[1:],compute_uv=False),'residual singulars',np.linalg.svd(R,compute_uv=False));print('projected residuals',R@np.linalg.inv(C[1:]))
print('cubic coeff residual',fit(np.c_[F,u**3],v)[5:]);print('quartic coeff residual',fit(np.c_[F,u**4],v)[5:])
funcs={'cube':u**3,'quartic':u**4,'abs':abs(u),'abs2':u*abs(u),'crossnext':u*np.roll(u,-1,1),'crossprev':u*np.roll(u,1,1),'next2':u*np.roll(u,2,1),'cube_next':u*u*np.roll(u,-1,1),'cube_prev':u*u*np.roll(u,1,1),'sin':np.sin(2*u),'cos':np.cos(2*u),'exp':np.exp(u),'den':(u+.45*u*u)*u*u}
for lab,G in funcs.items():
 for j in [0,1]:
  if j==0: f=lambda k:aff(u+k[0]*u*u+k[1]*G)
  else: f=lambda k:aff((u+k[0]*u*u)/(1+k[1]*G))
  def res(k):
   D=f(k);return (D@fit(D,v)-v).ravel()
  o=least_squares(res,[.45,0],max_nfev=100);print(lab,j,o.x,np.mean(o.fun**2)**.5)
for fold in [np.arange(56,72),*np.array_split(np.random.default_rng(25).permutation(n),5)]:
 ix=np.setdiff1d(np.arange(n),fold)
 d=base['dot']; m=base['mix']; D=aff(np.c_[a,b,d]); Ds=np.c_[b,m]; Df=np.c_[a,d]
 def profile(k,ix,details=False):
  tt=T(k[:2]); P=np.c_[ (Ds[ix,None,:]*(tt[None,:,0:1]+k[2]*tt[None,:,1:2])).reshape(-1,5), (Df[ix,None,:]*tt[None,:,1:2]).reshape(-1,5) ]; coeff=fit(P,Y[ix].ravel())
  if details:return coeff
  return P@coeff-Y[ix].ravel()
 o=least_squares(profile,[.4,1.8,-.1],args=(ix,),bounds=([.1,.9,-1],[.89,4,1]));cc=profile(o.x,ix,True);tt=T(o.x[:2]); Z=np.c_[Ds[fold]@cc[:5], Df[fold]@cc[5:]+o.x[2]*(Ds[fold]@cc[:5])]@tt.T
 print('CV',o.x,'y',np.mean(abs(D[fold]@fit(D[ix],y[ix])-y[fold])),'traj',np.mean(abs(Z-Y[fold])),'traintraj',np.mean(abs(o.fun)))
