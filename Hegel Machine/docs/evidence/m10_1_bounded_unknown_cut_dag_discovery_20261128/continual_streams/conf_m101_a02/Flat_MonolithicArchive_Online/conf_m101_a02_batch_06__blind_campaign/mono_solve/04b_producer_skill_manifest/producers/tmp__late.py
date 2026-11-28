from operators import *
from scipy.linalg import solve
from scipy.optimize import minimize_scalar
from itertools import product
b,c,d=[ops[k](z) for k in 'BCD']; P=Y[:,1:]; y=Y[:,0]; az=z*(1-1.1*np.exp(-z*z)); n=len(y)
def scale(a):return (a-a[:n].mean(0))/a[:n].std(0)
def gram(a):
 a=scale(a); return a@a.T/a.shape[1]
Z,B,C,D,A=map(gram,(z,b,c,d,az)); L=np.c_[aff(z),b,c,d];
fits=[]
for i,j in folds:
 Q=np.linalg.qr(L[i])[0]; R=P[i]-Q@(Q.T@P[i]);
 def h(rate):return -np.expm1(-rate*T)
 def loss(rate):
  v=h(rate);return np.linalg.norm(R-np.outer(R@v/(v@v),v))
 rate=minimize_scalar(loss,bounds=(.1,4),method='bounded').x; v=h(rate); amp=P@v/(v@v); fits.append((i,j,amp));
 print('rate',rate,'residual std',np.std(R-np.outer(R@v/(v@v),v)))
for nm,F in [('B',aff(b)),('ZB',np.c_[aff(z),b]),('ZBC',np.c_[aff(z),b,c]),('ZBD',np.c_[aff(z),b,d]),('ZBCD',L),('ZAcos',np.c_[aff(z),az,np.cos(1.268*z)]),('ZAcosQ',np.c_[aff(z),az,np.cos(1.268*z),pol(z,2)[:,5:]]),('ZBCDq',np.c_[L,pol(b,2)[:,5:]])]:
 print('contrast',nm,[(round(np.mean(abs(F[j]@w[:-1]+amp[j]*w[-1]-y[j])),4),round(w[-1],3)) for i,j,amp in fits for w in [fit(np.c_[F[i],amp[i]],y[i])]])
extra={'z':Z,'b':B,'c':C,'d':D,'lin':Z+B+C+D,'none':0,'b2':B**2,'z2':Z**2,'zb':Z*B,'zd':Z*D,'bd':B*D,'rbf':np.exp(-1.4*(np.diag(Z)[:,None]+np.diag(Z)[None,:]-2*Z)),'smooth':gram(np.c_[np.tanh(1.7*z),np.sin(1.7*z),np.cos(1.27*z)]),'mix':Z*B+Z*D+B*D,'squares':gram(np.c_[z*z,b*b]),'abc':gram(np.c_[np.sin(1.7*b),np.tanh(1.7*b),b*b])}
# Trajectory and label contrast can have different spatial regularity.
kt0=1+B+.3*B**2+.3*(Z+C+D)+2*Z*B
kt1=1+B+.3*B**2+Z+2*Z*B
trials=[]; baselines=[]
for kt in [kt0,kt1,1+B+.3*B**2+.3*(Z+C+D)+2*Z*B+.3*B**3,1+B+.3*B**2+.3*(Z+C+D)+2*Z*B+.3*Z*Z]:
 for lam in (.0001,.001,.005,.02):
  amps=[kt[np.ix_(j,i)]@solve(kt[np.ix_(i,i)]+lam*np.eye(len(i)),u[i],assume_a='pos') for i,j,u in fits]; baselines.append(amps)
for name,E in extra.items():
 for w,lam in product((.03,.3,2),(.0001,.001,.01,.06)):
  K=1+Z+B+.25*(C+D)+w*E
  H=[K[np.ix_(j,i)]@solve(K[np.ix_(i,i)]+lam*np.eye(len(i)),np.eye(len(i)),assume_a='pos') for i,j,u in fits]
  for k,at in enumerate(baselines):
   for cf in (.7,.9,1,1.1,1.3):
    e=np.array([np.mean(abs(hh@(y[i]-cf*u[i])+cf*ap-y[j])) for hh,ap,(i,j,u) in zip(H,at,fits)]); trials.append((e,name,w,lam,k,cf))
for key in (lambda x:x[0].max(),lambda x:x[0][0]):
 print('SHARED',[(a.round(5).tolist(),*p) for a,*p in sorted(trials,key=key)[:12]])
# Test if residual is a single ridge term in observed or augmented coordinates.
for k,x in [('z',z),('b',b),('a',az),('c',c),('d',d),('zb',np.c_[z,b]),('zd',np.c_[z,d])]:
 for tag in ('sin','tanh','cos','rational'):
  errors=[]; dirs=[]
  for i,j,u in fits:
   base=np.c_[L[:n],u]; Q=np.linalg.qr(base[i])[0]; ry=y[i]-Q@(Q.T@y[i]); a=x[i];
   func={'sin':np.sin,'cos':np.cos,'tanh':np.tanh,'rational':lambda t:1/(1+t*t)}[tag]
   def rr(p):
    f=func(a@p[:-2]+p[-2]); return ry-p[-1]*(f-Q@(Q.T@f))
   rng=np.random.default_rng(718); cand=[least_squares(rr,np.r_[rng.normal(0,.75,a.shape[1]),rng.normal(0,.4),.1],max_nfev=110) for _ in range(12)]; best=min(cand,key=lambda r:np.linalg.norm(r.fun)); p=best.x; f=func(x@p[:-2]+p[-2])*p[-1]; w=fit(base[i],y[i]-f[i]); obs=np.mean(abs(base[j]@w+f[j]-y[j])); errors.append(obs); dirs.append(p.round(2));
  print('RIDGE',k,tag,np.round(errors,4),dirs[0])
