exec(open('/tmp/grammar6.py').read().split('for pk,names')[0])
from scipy.optimize import least_squares
from scipy.special import erf
N=np.arange(len(x)); folds=[N[52:]]+list(np.array_split(np.random.default_rng(17).permutation(N),6))+[j for k in range(4) for j in (np.argsort(x[:,k])[:12],np.argsort(x[:,k])[-12:])]
np.set_printoptions(precision=4,suppress=True)
warps={'sin3':lambda z,a:np.sin(a*z)**3,'tanh3':lambda z,a:np.tanh(a*z)**3,'rat3':lambda z,a:z**3/(1+a*z*z),'rat5':lambda z,a:z**3/(1+a*z**4),'rat32':lambda z,a:z**3/(1+a*z*z)**2,'exp3':lambda z,a:z**3*np.exp(-a*z*z),'root3':lambda z,a:z**3/(1+a*z*z)**1.5,'cubicsoft':lambda z,a:z**3/np.sqrt(1+a*z**6)}
def design(z,w,pair):
 return np.c_[one(z),w,z,w*w,q(w,pairs) if pair==0 else q(z,pairs) if pair==1 else q(w,pairs)+.1*q(z,pairs)]
def fit(name,op,reg,mode,i,j):
 Z=x[i]; U=Y[i]; scale=U.std(0); wf=warps[name];
 def mats(p,z):
  A=design(z,wf(z,p[0] if mode==0 else p[:4]),op);return A
 na=1 if mode==0 else 4
 def fitcoef(A):
  if not reg:return np.linalg.lstsq(A,U,rcond=None)[0]
  d=A.std(0);d[0]=1.; d[1:9]*=.15;d[0]=0.; return np.linalg.solve(A.T@A+reg*np.diag(d*d),A.T@U)
 def obj(p):
  A=mats(p,Z); r=((A@fitcoef(A)-U)/scale).ravel();return np.r_[r,(p-p.mean())*1.0] if mode else r
 default={'sin3':.9,'tanh3':.75,'rat3':.75,'rat5':.25,'rat32':.35,'exp3':.25,'root3':.45,'cubicsoft':.5}[name];p=least_squares(obj,np.full(na,default),bounds=(.001,3.),max_nfev=35).x
 W=fitcoef(mats(p,Z)); _,_,vt=np.linalg.svd(mats(p,Z)@W[:,1:],full_matrices=False);W[:,1:]=(W[:,1:]@vt[:2].T)@vt[:2];e=abs(mats(p,x[j])@W-Y[j]);return [e[:,0].mean(),e[:,1:].mean(),e[:,1:].mean(0).max()],p
out=[]
for n in warps:
 for op in (0,1,2):
  for reg in (0,.6,2,6):
   for mode in (0,1):
    e=np.array([fit(n,op,reg,mode,np.setdiff1d(N,j),j)[0] for j in folds]);out.append((e[7:,:2].max(),n,op,reg,mode,e[0],e[:7].max(0),e[7:].max(0)))
for row in sorted(out,key=lambda r:r[0])[:32]:print(*row)
print('best per family');
for n in warps: print(*min((r for r in out if r[1]==n),key=lambda r:r[0]))
