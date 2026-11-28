import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 from probe6 import *
from scipy.special import expit
V=c['v2']; q=lat['v1cross'];
forms={'square':lambda z,a:z+a*z*z,'square_cube':lambda z,a,b:z+a*z*z+b*z**3,'square4':lambda z,a,b:z+a*z*z+b*z**4,'poly4':lambda z,a,b,k:z+a*z*z+b*z**3+k*z**4,'quotient':lambda z,a,b:(z+a*z*z)/(1+b*z*z),'ratlin':lambda z,a:z/(1-a*z),'tanh_square':lambda z,a,b:np.tanh(b*z)/b+a*z*z,'sin_square':lambda z,a,b:np.sin(b*z)/b+a*z*z,'exp':lambda z,a:np.expm1(a*z)/a,'shifted_sqrt':lambda z,a,b:z+a*(np.sqrt(1+b*z*z)-1)}
def residual(par,f,u=cu,v=V):
 d=A(f(u,*par));return (d@L(d,v)-v).ravel()
import inspect
for k,f in forms.items():
 fit2=least_squares(residual,[.4]+[.1]*(len(inspect.signature(f).parameters)-2),args=(f,),max_nfev=500);print('C2',k,fit2.x,np.std(fit2.fun))
for j in range(4):
 d=np.column_stack([cu[:,j]**i for i in range(1,7)]); oth=np.delete(cu,j,axis=1); oth=oth+.45*oth**2; D=np.c_[A(oth),d]; M=L(D,V); print('C2 powers',j,M[-6:].T)
f=forms['poly4']; par=least_squares(residual,[.4,.1,.1],args=(f,)).x
D=A(f(cu,*par)); coef=L(D,V)
def C2(z):return A(f(z,*par))@coef
C1=lambda z:cross(z)@L(cross(cu),c['v1'])
r=C2(w); print('range',[(a.min(0),a.max(0)) for a in (w,q,r)]);
def score(d,Y=Y):
 e=d[52:]@L(d[:52],Y[:52])-Y[52:]; t=d[:52]@L(d[:52],Y[:52])-Y[:52]; return tuple(np.round((np.mean(abs(t[:,0])),np.mean(abs(t[:,1:])),np.mean(abs(e[:,0])),np.mean(abs(e[:,1:]))),5))
for k,z0 in [('q',q),('r',r),('q+r',q+r),('q-r',q-r),('q*r',q*r),('q/r',q/(1+r**2)),('q+r*q',q+r*q),('r+q*r',r+q*r),('gate',q*expit(r)),('blend',q*expit(r)+r*expit(-r)),('C2-after-C1',C2(q)),('C1-after-C2',C1(r)),('C2-after-A',C2(z)),('C1-B-C2-A',C1(B(C2(z)))),('q,r',np.c_[q,r]),('q,w',np.c_[q,w]),('q,r,q*r',np.c_[q,r,q*r]),('q,r,cross',np.c_[q,r,*[q[:,i]*r[:,j] for i in range(4) for j in range(4)]])]:
 print('branch',k,score(A(z0)),score(poly(z0,2)) if z0.shape[1]<=8 else '',score(np.c_[A(z0),z0[:,0]*z0[:,2],z0[:,1]*z0[:,3]]) if z0.shape[1]==4 else '')
print('train trajectory fit shared time modes');print(np.linalg.svd(Y[:52,1:],full_matrices=False)[2][:3]);print('center',np.mean(Y[:52],axis=0))
