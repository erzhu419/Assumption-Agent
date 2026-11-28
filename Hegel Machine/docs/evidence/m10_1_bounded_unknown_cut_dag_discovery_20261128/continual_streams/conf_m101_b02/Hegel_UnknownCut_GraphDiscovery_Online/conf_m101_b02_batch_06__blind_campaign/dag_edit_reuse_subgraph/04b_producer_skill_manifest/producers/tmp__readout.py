from probe import *
from scipy.linalg import svd
rng=np.random.default_rng(78)
def fit(a,y):return np.linalg.lstsq(a,y,rcond=1e-9)[0]
def cv(a,Y):
 a=np.c_[np.ones(len(a)),a]; out=[]
 for seq in [np.arange(72),rng_perm]:
  e=[]
  for idx in np.array_split(seq,4):
   m=np.ones(72,bool);m[idx]=False;e.append(np.mean(abs(a[idx]@fit(a[m],Y[m])-Y[idx]),axis=0))
  out.append(np.max(e,axis=0))
 return out
rng_perm=rng.permutation(72)
def c1f(x):return np.c_[x[:,0]*x[:,1],x[:,1]*x[:,2],x[:,2]*x[:,3],x[:,3]*x[:,0]]
def c2f(x):return np.c_[x,x*x]
c=lambda x: np.c_[np.ones(len(x)),c1f(x)]@fit(np.c_[np.ones(len(u)),c1f(u)],v1)
d=lambda x: np.c_[np.ones(len(x)),c2f(x)]@fit(np.c_[np.ones(len(u)),c2f(u)],v2)
np.set_printoptions(precision=4,suppress=True)
U,S,V=svd(tr,full_matrices=False);print('S',S,'V',V[:3]); print('mean',tr.mean(0))
for name,F in {'z':z,'zb':np.c_[z,b], 'zbc':np.c_[z,b,c(z)],'zbcd':np.c_[z,b,c(z),d(z)],'zbCb':np.c_[z,b,c(b)],'zbCbDb':np.c_[z,b,c(b),d(b)],'zbczCb':np.c_[z,b,c(z),c(b)],'zbc+nonlin':np.c_[z,b,c(z),np.tanh(2*b),np.tanh(2*c(z))], 'zbcd+nonlin':np.c_[z,b,c(z),d(z),np.tanh(2*b),np.tanh(2*c(z)),np.tanh(2*d(z))]}.items():
 Y=np.c_[y,U[:,:3]*S[:3]];f=np.c_[np.ones(len(z)),F];r=Y-f@fit(f,Y);print(name,'fit',np.mean(abs(r),0),'cv',cv(F,Y)[0]);
 if name in ('zbcd','zbCbDb'):print('coeff pca',fit(f,Y))
# Identify shared time modes by fitting the matrix, not independently estimating a rate per noisy row.
for name,fun,p0,lo,hi in [
 ('exp',lambda k:np.c_[-np.expm1(-k[0]*t)],[.6],[.02],[5]),
 ('exp + const',lambda k:np.c_[-np.expm1(-k[0]*t),np.ones(len(t))],[.6],[.02],[5]),
 ('exp + drift',lambda k:np.c_[-np.expm1(-k[0]*t),t],[.6],[.02],[5]),
 ('exp + exp',lambda k:np.c_[-np.expm1(-k[0]*t),-np.expm1(-k[1]*t)],[.4,1.6],[.02,.02],[8,8]),
 ('exp + sin',lambda k:np.c_[-np.expm1(-k[0]*t),np.sin(k[1]*t)],[.6,.9],[.02,.02],[5,4]),
 ('exp + drift + const',lambda k:np.c_[-np.expm1(-k[0]*t),t,np.ones(len(t))],[.6],[.02],[5]),
 ('exp + exp + const',lambda k:np.c_[-np.expm1(-k[0]*t),-np.expm1(-k[1]*t),np.ones(len(t))],[.4,1.6],[.02,.02],[8,8]),
 ('exp + exp + drift',lambda k:np.c_[-np.expm1(-k[0]*t),-np.expm1(-k[1]*t),t],[.4,1.6],[.02,.02],[8,8]),
 ('exp + sin + drift',lambda k:np.c_[-np.expm1(-k[0]*t),np.sin(k[1]*t),t],[.6,.9],[.02,.02],[5,4]),
 ('exp + sin + cos',lambda k:np.c_[-np.expm1(-k[0]*t),np.sin(k[1]*t),np.cos(k[1]*t)-1],[.6,.9],[.02,.02],[5,4])]:
 def res(k):
  h=fun(k);return (h@fit(h,tr.T)-tr.T).ravel()
 k=least_squares(res,p0,bounds=(lo,hi)).x;H=fun(k);co=fit(H,tr.T).T
 print(name,k,np.sqrt(np.mean(res(k)**2)), 'amp corr y',np.corrcoef(np.c_[y,co].T)[0,1:])
