from compose import *
from scipy.optimize import root
u=U[:,1:]; g=np.sort(ids); v=V[g];
soft=lambda a:a/(1+abs(a))
def fitA():
 def features(z,j,p):
  w=z[:,j]; r=z[:,(j+2)%4]; return one(np.c_[w,soft(p[0]*w),r,np.tanh(p[1]*r)])
 def residual(p):
  return np.concatenate([D@ls(D,v[:,j])-v[:,j] for j in range(4) for D in [features(u[g],j,p)]])
 p=least_squares(residual,[1,1],bounds=([.08,.08],[8,8]),gtol=1e-13).x
 P=[ls(features(u[g],j,p),v[:,j]) for j in range(4)]; print('A nonlinear',p,np.std(residual(p)),P)
 return lambda z:np.column_stack([features(z,j,p)@P[j] for j in range(4)])
FA=fitA(); az=np.array([root(lambda z:FA(z[None])[0]-r,(r-M[0])@np.linalg.inv(M[1:])).x for r in x]); print('inv max',abs(FA(az)-x).max(),az.min(0),az.max(0))
if __name__=='__main__':
 F=dict(left=az, right=one(az)@B, rinv=(az-B[0])@np.linalg.inv(B[1:]),raw=x)
 for k,z in F.items():
  c,d=[f(z) for f in branches]; m=c*d; cross=(c[:,:,None]*d[:,None,:]).reshape(n,-1); norm=lambda r:np.sum(r*r,axis=1)
  for l,D in [('lin',np.c_[c,d]),('dot',np.c_[c,d,m.sum(1)]),('diag',np.c_[c,d,m]),('diag+',np.c_[c,d,m,norm(c),norm(d),norm(z)]),('cross',np.c_[c,d,cross]),('input',np.c_[c,d,m,z]),('mix',np.c_[c,d,m,cross,z]),('pquad',np.c_[z,*[z[:,i]*z[:,j] for i in range(4) for j in range(i,4)]])]: print(k,l,np.round(score(one(D)),6))
