exec(open('/tmp/readouts05.py').read().split('sol=[]')[0])
from itertools import combinations
np.set_printoptions(precision=7,suppress=True)
p0=[.57257,2.265422,-.091598,.008623,9.917495,.003997,13.274296]
def feature(x,p):
 k,power,l=p[:3]; g=x**3/(1+k*np.abs(x)**power)+l*x
 for a,f in np.reshape(p[3:],(-1,2)):g+=a*np.sin(f*x)
 return aff(g)
def panel(p):
 a=feature(U,p); op=ls(a,V); return a@op-V,op
p2=least_squares(lambda p:panel(p)[0].ravel(),p0,max_nfev=400).x
r,op=panel(p2); print('two',p2,'rms',np.std(r),'svd',np.linalg.svd(r,compute_uv=False)); print('coordinate inverse residual',np.std(r@np.linalg.inv(op[1:]),axis=0))
lo=[.001,.5,-2]; hi=[30,5,2]
sol=[]
for f in np.arange(2.5,34,1):
 p=np.r_[p2,.001,f]; n=len(p[3:])//2
 o=least_squares(lambda p:panel(p)[0].ravel(),p,bounds=(lo+[-.25,1]*n,hi+[.25,40]*n),max_nfev=350)
 sol.append(o)
sol.sort(key=lambda o:o.cost)
for o in sol[:6]: print('three',np.std(panel(o.x)[0]),o.x)
for f in np.arange(3,34,2):
 p=np.r_[sol[0].x,.001,f]; n=len(p[3:])//2
 o=least_squares(lambda p:panel(p)[0].ravel(),p,bounds=(lo+[-.25,1]*n,hi+[.25,40]*n),max_nfev=350)
 print('four',np.std(panel(o.x)[0]),o.x)
# Determine whether the residual is a shared separable curve or mixes coordinates.
z=feature(U,sol[0].x); residual,op=panel(sol[0].x); print('three inverse residual',np.std(residual@np.linalg.inv(op[1:]),axis=0));
for j in range(1,4):
 w=np.roll(U,-j,axis=1)
 for name,g in [('product',U*w),('cubicmix',U*w*w),('sinemix',np.sin(U+w)),('rationalmix',U**3/(1+w*w)),('sin11',np.sin(11*(U+w)))]:
  d=np.c_[z,g]; print('cross',j,name,np.std(d@ls(d,V)-V))
