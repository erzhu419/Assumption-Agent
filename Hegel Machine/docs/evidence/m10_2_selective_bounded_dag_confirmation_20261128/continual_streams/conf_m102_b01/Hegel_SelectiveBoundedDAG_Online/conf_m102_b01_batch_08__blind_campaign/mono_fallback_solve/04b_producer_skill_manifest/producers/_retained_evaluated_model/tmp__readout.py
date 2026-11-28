from base import *
from scipy.linalg import svd
C=m._pairmap(bs['sin+cbrt'])@C0; B=bs['sin+cbrt']; N=len(Y)
Q,S,V=np.linalg.svd(Y[:,1:]-Y[:,1:].mean(0),full_matrices=False); print('temporal',S,V[:3]); print('y correlation PC',np.corrcoef(np.c_[Y[:,0],Q[:,:6]].T)[0]); print('y projection',np.linalg.lstsq(aff(Q[:,:3]),Y[:,0],rcond=None)[0]);
splits=[]
for a in (52,0,20,40):splits.append(np.arange(a,a+20))
for seed in range(8):splits.append(np.random.default_rng(seed).permutation(72)[:20]);

def status(P):
 return np.round([P[0],P[:4].mean(0),P[:4].max(0),P[4:].mean(0),P[4:].max(0)],4).tolist()
def kscore(K,reg):
 vals=[]
 for te in splits:
  tr=np.setdiff1d(np.arange(N),te); mu=Y[tr].mean(0)
  e=abs(K[np.ix_(te,tr)]@np.linalg.solve(K[np.ix_(tr,tr)]+reg*np.eye(len(tr)),Y[tr]-mu)+mu-Y[te]);vals.append([e[:,0].mean(),e[:,1:].mean()]);
 return np.array(vals)
def gram(*a):
 F=np.column_stack(a); F=(F-F.mean(0))/np.maximum(F.std(0),1e-6); return F@F.T/F.shape[1]
def cross(c,d):return (c[:,:,None]*d[:,None,:]).reshape(N,-1)
def cen(k):return k-k.mean(0)[None,:]-k.mean(1)[:,None]+k.mean()
results=[]
for an,fun in [('atan',np.arctan),('asinh',np.arcsinh),('tanh',np.tanh),('linear',lambda a:a)]:
 for gain in (.6,1.25,2.0):
  c=fun(gain*C); d=np.tanh(1.25*D); main=gram(c,d); products=gram(cross(c,d)); diag=gram(c*d); b=gram(B); sq=gram(c**2,d**2); cp=gram(cross(c,c)); dp=gram(cross(d,d));
  for kind,k in {'cross':main+products,'diag':main+diag,'mix':main+diag+products,'b01':main+products+.1*b,'b03':main+products+.3*b,'b1':main+products+b,'diagb':main+diag+.3*b,'hybrid':main+products+.12*sq+.12*b,'extended':main+products+.2*cp+.2*dp+.3*b,'poly3':main+products+.1*cen(main**2)+.07*cen(main**3)+.3*b,'resid':main+products+.2*cen(products*main)+.3*b}.items():
   for reg in (.001,.008,.025,.07):
    p=kscore(k+10,reg); results.append((p[:4].mean(0)+.25*p[:4].max(0)+.5*p[4:].mean(0),an,gain,kind,reg,p))
for j in [0,1]:
 print('RANK',j)
 for r in sorted(results,key=lambda r:r[0][j])[:15]:print(r[1:-1],status(r[-1]))
