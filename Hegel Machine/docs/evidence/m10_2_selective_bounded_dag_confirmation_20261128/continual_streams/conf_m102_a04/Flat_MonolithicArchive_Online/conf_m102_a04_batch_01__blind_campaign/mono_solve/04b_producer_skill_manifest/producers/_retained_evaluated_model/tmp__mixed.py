import contextlib,io
with contextlib.redirect_stdout(io.StringIO()): from probe import *
from sklearn.preprocessing import PolynomialFeatures
from scipy.linalg import svd
np.set_printoptions(precision=5,suppress=True,linewidth=180)
def prod(x):return np.c_[x[:,0]*x[:,1],x[:,0]*x[:,3],x[:,1]*x[:,2],x[:,2]*x[:,3]]
U=prod(u); C=fit(one(U),c['v1']); a1=one(prod(x))@C; a2=maps['v2']; print('C1 matrix',C,'noise',np.std(one(U)@C-c['v1'])); p=.2696617; c2f=lambda x:one(np.c_[x,x*x,x*x/(1+p*x*x)]); D=fit(c2f(u),c['v2']); print('C2 matrix',D,'Ratios',D[5:9]/D[1:5],D[9:]/D[1:5]);
P=PolynomialFeatures(3); F=P.fit_transform(x); cf=fit(F[:52],Y[:52]); print('cubic coefficients',list(zip(P.get_feature_names_out(),cf.round(4)))); print('cubic std',np.std(F[:52]@cf-Y[:52],axis=0))
def report(name,F):
 F=one(F); co=fit(F[:52],Y[:52]); err=F[52:]@co-Y[52:]; print(name,len(co),np.round(np.std(F[:52]@co-Y[:52],axis=0),5),np.round(np.r_[np.mean(abs(err),0),np.mean(abs(err[:,1:]@T.T))],5)); return co
mul=lambda a,b:(a[:,:,None]*b[:,None,:]).reshape(len(a),-1)
for name,z in [('x',x),('B',b),('c2',a2)]:
 for quad in [False,True]:
  f=np.c_[a1,z,z*z] if quad else np.c_[a1,z]; report(name+str(quad),f); report(name+str(quad)+' int',np.c_[f,mul(a1,z)]); report(name+str(quad)+' diag int',np.c_[f,a1*z]);
for z, name in [(a1,'c1'),(a2,'c2'),(x,'x'),(b,'b')]:
 for kind,fn in [('sin',lambda z:np.sin(1.25*z)),('tanh',lambda z:np.tanh(1.25*z)),('rat2',lambda z:z/(1+.22*z*z)),('ratabs',lambda z:z/(1+.22*abs(z))),('sq',lambda z:z*z/(1+.22*z*z))]:
  f=np.c_[a1,a2,fn(z)]; report(name+' '+kind,f)
for q in [0,.1,.22,.4,1]:
 for k,zz in [('x',x),('a1',a1),('c2',a2),('b',b)]:
  # Per-coordinate quotients of mixed interactions, including a common norm gate.
  rr=1/(1+q*zz**2); sf=np.c_[a1,a2,a1*rr,a2*rr,a1/(1+q*np.sum(zz**2,axis=1,keepdims=True))]; report('quot '+k+str(q),sf)
